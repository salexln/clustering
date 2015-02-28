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
import org.apache.spark.SparkContext._

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
  def getFuzzySets(data: RDD[Vector]) = {
    //val breezeData = formatData(data)
    //breezeData.foreach { x => println(x) }
    val norms = data.map(v => breezeNorm(v.toBreeze, 2.0))
    norms.persist()
    val breezeData = data.map(_.toBreeze).zip(norms).map {
      case (v, norm) =>
        new BreezeVectorWithNorm(v, norm)
    }

    val sc = breezeData.sparkContext
    val dimNumb = data.first().size
    val broadcastCenters = sc.broadcast(centroids)
    val broadcastCorrection = sc.broadcast(EPSILON)
    val broadcastDim = sc.broadcast(dimNumb)

    val mapper = breezeData.mapPartitions { points =>
      val pointsCopy = points.duplicate
      val nPoints = pointsCopy._1.length      
      val singleDist = Array.fill[Double](c)(0)
      val numDist = Array.fill[Double](c)(0)
      val membershipMatrix = Array.ofDim[Double](nPoints, c)
      val datums = Array.fill(nPoints)(BDV.zeros[Double](broadcastDim.value).asInstanceOf[BV[Double]])
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
       *  var sign will be p (positive) if the datum's feature is greater (above) the cluster's prototype for
       *  that feature otherwise sign will be n (negative)
       */
      val outMapper = for (prototN <- 0 until c; elem <- 0 until nPoints; feature <- 0 until broadcastDim.value) yield {
        var sign = "p"
        if (datums(elem)(feature) >= broadcastCenters.value(prototN).vector(feature)) { sign = "p" }
        else { sign = "n" }
        ((prototN, feature, sign), (datums(elem)(feature), membershipMatrix(elem)(prototN)))
      }
      outMapper.iterator
    }.filter(f => f._2._2 > treshold)

    /**
     * Put axis' origin into centroids' point for each feature
     *  Remember that each center belongs to his cluster with degree 1
     */
    val traslatedPoint = mapper.map { f =>
      val c = ((f._1._1, f._1._2, f._1._3), ( (f._2._1 - broadcastCenters.value(f._1._1).vector(f._1._2)), (f._2._2 - 1)) )
      c
    }

    /**
     * Linear Regression without intercept y = bx. Given a set of points:
     * (x(i),y(i)) => b = sum(x(i) * y(i) / sum(x(i))
     * Note that x(i) = features, y(i) = membership degree
     * The linear regression will be computed traslating the points around (x0,y0)
     *  so the regression will fit the model y-y0 = b(x - x0)
     */
    val bNum = traslatedPoint.map { f =>
      val c = ((f._1._1, f._1._2, f._1._3), (f._2._1 * f._2._2))
      c
    }.reduceByKey((x, y) => x + y)

    val bDen = traslatedPoint.map { f =>
      val c = ((f._1._1, f._1._2, f._1._3), f._2._1)
      c
    }.reduceByKey((x, y) => x + y)

    val b = bNum.join(bDen).map { f =>
      val c = ((f._1._1, f._1._2, f._1._3), (f._2._1 / (f._2._2 + broadcastCorrection.value)))
      c
    }
    // Rember that the equation is y-y0=b(x-x0) so y=0 => x=(bx0-y0)/b
    val intersections = b.map { f =>
      val aux = (( (f._2 * broadcastCenters.value(f._1._1).vector(f._1._2)) - 1) / f._2)
      val c = ((f._1._1, f._1._2, f._1._3), aux)
      c
    }

   // val joinedInters = intersections.join(intersections)

    //joinedInters.sortBy(f => f._1._1, true).sortBy(f=>f._1._2,true)
    val output = intersections.collectAsMap().toSeq.sortBy(f => f._1._1).sortBy(f => f._1._2)
    output.foreach(f=>println(f))

    broadcastCenters.destroy(true)
    broadcastCorrection.destroy(true)
    broadcastDim.destroy(true)
    
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
