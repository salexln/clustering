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

import org.apache.spark.api.java.JavaRDD
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import breeze.linalg.{ DenseVector => BDV, Vector => BV, norm => breezeNorm }

class FuzzyCMeansModel(centroids: Array[BreezeVectorWithNorm],
                       m: Double) extends Serializable {

  /** @param c: Total number of clusters. */
  private def c: Int = centroids.length
  /**
   * @param treshold: data pre-filter.
   *  Feed the reducer cutting out datums with small memebership degree
   *  to the j-th  prototype
   */
  private def treshold = 0.1

  /** Print the clusters center. */
  def Centers() {
    println("Cluster Centers")
    for (i <- 0 until c) {
      println(centroids.apply(i).vector)
    }
  }

  /**
   * Compute fuzzy sets
   */
  def getU(data: RDD[Vector]) = {
    //val breezeData = formatData(data)
    //breezeData.foreach { x => println(x) }
    val norms = data.map(v => breezeNorm(v.toBreeze, 2.0))
    norms.persist()
    val breezeData = data.map(_.toBreeze).zip(norms).map {
      case (v, norm) =>
        new BreezeVectorWithNorm(v, norm)
    }
    //val indexedData = breezeData.zipWithIndex()
    val sc = breezeData.sparkContext
    val dimNumb = data.first().size
    val broadcastCenters = sc.broadcast(centroids)
    val broadcastCorrection = sc.broadcast(EPSILON)

    val mapper = breezeData.mapPartitions { points =>
      val pointsCopy = points.duplicate
      val nPoints = pointsCopy._1.length
      val singleDist = Array.fill[Double](c)(0)
      val numDist = Array.fill[Double](c)(0)
      val membershipMatrix = Array.ofDim[Double](nPoints, c)
      val datums = Array.fill(nPoints)(BDV.zeros[Double](dimNumb).asInstanceOf[BV[Double]])
      var i = 0

      pointsCopy._2.foreach { point =>
        var denom = 0.0
        for (j <- 0 until c) {
          singleDist(j) = (KMeans.fastSquaredDistance(point, broadcastCenters.value(j)) +
            broadcastCorrection.value)
          numDist(j) = math.pow(singleDist(j), (1 / (m - 1)))
          denom += (1 / numDist(j))
        }
        for (j <- 0 until c) {
          val u = (numDist(j) * denom) //uij^m  
          membershipMatrix(i)(j) = (1 / u)
        }
        datums(i) = point.vector
        i += 1
      }

      /**
       *  Split elements around cluster's prototype
       *  var sign will be 0 if the datum's feature is greater (above) the cluster's prototype for
       *  that feature
       */
      val outMapper = for (prototN <- 0 until c; elem <- 0 until nPoints; feature <- 0 until dimNumb) yield {
        var sign = "p"
        if (datums(elem)(feature) >= broadcastCenters.value(prototN).vector(feature)) { sign = "p" }
        else { sign = "n" }
        ((prototN, feature, sign), (membershipMatrix(elem)(prototN), datums(elem)(feature)))
      }
      outMapper.iterator
    }.filter(f => f._2._1 > treshold)
    

    
  }

  /**
   * Helper methods for lazy evaluation of a factor EPSILON
   * to avoid division by zero while computing membership degree
   * for each datum to each center
   */
  private lazy val EPSILON = {
    var eps = 1.0
    while ((1.0 + (eps / 2.0)) != 1.0) {
      eps /= 2.0
    }
    eps
  }

  private def formatData(data: RDD[Vector]): RDD[BreezeVectorWithNorm] = {
    // Compute squared norms and cache them.
    val norms = data.map(v => breezeNorm(v.toBreeze, 2.0))
    norms.persist()
    val breezeData = data.map(_.toBreeze).zip(norms).map {
      case (v, norm) =>
        new BreezeVectorWithNorm(v, norm)
    }
    breezeData
  }

  /**
   * Retrieve Centers' Vectors
   */
  def getCenters(): Array[Vector] = {
    centroids.map { c => Vectors.fromBreeze(c.vector) }
  }

}
