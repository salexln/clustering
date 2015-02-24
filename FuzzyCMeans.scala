/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.clustering

import scala.collection.mutable.ArrayBuffer

import breeze.linalg.{ DenseVector => BDV, Vector => BV, norm => breezeNorm }
import org.apache.spark.annotation.Experimental
import org.apache.spark.Logging
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import scala.util.Random
import org.apache.spark.util.random.XORShiftRandom

class FuzzyCMeans private (
  private var c: Int,
  private var m: Double,
  private var maxIterations: Int,
  private var epsilon: Double) extends Serializable with Logging {

  /**
   * Constructs a Fuzzy CMeans instance with default parameters: {c: 2, m: 2, maxIterations: 20,
   * runs: 1, epsilon: 1e-4}.
   */
  def this() = this(2, 2, 20, 1e-4)

  /**
   * Helper methods already defined in ML lib. Used in FCM to
   * correct distances equals to 0
   */
  private lazy val EPSILON = {
    var eps = 1.0
    while ((1.0 + (eps / 2.0)) != 1.0) {
      eps /= 2.0
    }
    eps
  }

  /** Set the number of clusters to create (c). Default: 2. */
  def setC(c: Int): this.type = {
    this.c = c
    this
  }

  /** Set the value of the fuzziness constant (m). Default: 2. */
  def setM(m: Double): this.type = {
    this.m = m
    this
  }

  /** Set maximum number of iterations to run. Default: 20. */
  def setMaxIterations(maxIterations: Int): this.type = {
    this.maxIterations = maxIterations
    this
  }

  /**
   * Set the distance threshold within which we've consider centers to have converged.
   * If all centers move less than this Euclidean distance, we stop iterating one run.
   */
  def setEpsilon(epsilon: Double): this.type = {
    this.epsilon = epsilon
    this
  }

  /**
   * Train a K-means model on the given set of points; `data` should be cached for high
   * performance, because this is an iterative algorithm.
   */
  def run(data: RDD[Vector]): FuzzyCMeansModel = {
    // Compute squared norms and cache them.
    val norms = data.map(v => breezeNorm(v.toBreeze, 2.0))
    norms.persist()
    val breezeData = data.map(_.toBreeze).zip(norms).map {
      case (v, norm) =>
        new BreezeVectorWithNorm(v, norm)
    }
    val model = runBreeze(breezeData)
    norms.unpersist()
    model
  }

  /**
   * Train a Fuzzy C-means model on the given set of points; `data` should be cached for high
   * performance, because this is an iterative algorithm.
   */
  def runBreeze(data: RDD[BreezeVectorWithNorm]): FuzzyCMeansModel = {

    if (m <= 1) {
      throw new IllegalArgumentException("Invalid m: set a value of fuzziness greater than 1")
    }

    val initStartTime = System.nanoTime()
    val sc = data.sparkContext
    var centroids = initRandomCenters(data)
    val initTimeInSeconds = (System.nanoTime() - initStartTime) / 1e9
    logInfo(s"Initialization took " + "%.3f".format(initTimeInSeconds) + " seconds.")
    val dim = data.first().vector.length
    var convergence = false
    var iteration = 1
    val iterationStartTime = System.nanoTime()

    // Execute iterations of Fuzzy C Means algorithm 
    while (iteration < maxIterations && !convergence) {

      val broadcastCenters = sc.broadcast(centroids)
      val broadcastCorrection = sc.broadcast(EPSILON)

      val totContr = data.mapPartitions { points =>
        val partialNum = Array.fill(c)(BDV.zeros[Double](dim).asInstanceOf[BV[Double]])
        val partialDen = Array.fill[Double](c)(0)
        val singleDist = Array.fill[Double](c)(0)
        val numDist = Array.fill[Double](c)(0)

        points.foreach { point =>
          var denom = 0.0
          for (j <- 0 until c) {
            singleDist(j) = (KMeans.fastSquaredDistance(point, broadcastCenters.value(j)) + broadcastCorrection.value)
            numDist(j) = math.pow(singleDist(j), (2 / (m - 1)))
            denom += (1 / numDist(j))
          }
          for (j <- 0 until c) {
            val u = math.pow((numDist(j) * denom), -m) //uij^m            
            partialNum(j) += (point.vector * u) // local num of c(j) formula 
            partialDen(j) += u // local den of c(j) formula
          }
        }

        val centerContribs = for (j <- 0 until c) yield {
          (j, (partialNum(j), partialDen(j)))
        }
        centerContribs.iterator

      }.reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2)).collectAsMap()

      //Update Centers
      var changed = false
      for (z <- 0 until c) {
        if (totContr(z)._2 != 0) {
          val newCenter = new BreezeVectorWithNorm(totContr(z)._1 / totContr(z)._2)
          if (KMeans.fastSquaredDistance(newCenter, centroids(z)) > epsilon * epsilon) {
            changed = true
          }
          centroids(z) = newCenter
        }
      }

      if (changed) { iteration += 1 }
      else { convergence = true }

      broadcastCenters.destroy(true)
      broadcastCorrection.destroy(true)

    } // end while

    val iterationTimeInSeconds = (System.nanoTime() - iterationStartTime) / 1e9
    logInfo(s"Iterations took " + "%.3f".format(iterationTimeInSeconds) + " seconds.")

    if (iteration == maxIterations) {
      logInfo(s"Fuzzy C Means reached the max number of iterations: $maxIterations.")
    } else {
      logInfo(s"Fuzzy C Means converged in $iteration iterations.")
    }

    new FuzzyCMeansModel(centroids, m)
  }

  /**
   * Collect c random elements from the dataset and create an Array of BreezeVectorWithNorm
   * of c centers
   */

  def initRandomCenters(data: RDD[BreezeVectorWithNorm]): Array[BreezeVectorWithNorm] = {
    val seed = new XORShiftRandom().nextInt()
    val sample = data.takeSample(true, c, seed)
    sample
  }

}

/**
 * Top-level methods for calling Fuzzy C-means clustering.
 */
object FuzzyCMeans {

  /**
   * Trains a Fuzzy C-means model using the given set of parameters.
   *
   * @param data training points stored as `RDD[Array[Double]]`
   * @param c number of clusters
   * @param maxIterations max number of iterations
   */
  def train(
    data: RDD[Vector],
    c: Int,
    m: Double,
    maxIterations: Int,
    epsilon: Double): FuzzyCMeansModel = {
    new FuzzyCMeans().setC(c)
      .setM(m)
      .setMaxIterations(maxIterations)
      .setEpsilon(epsilon)
      .run(data)
  }

}
