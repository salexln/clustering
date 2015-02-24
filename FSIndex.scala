package org.apache.spark.mllib.clustering

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import breeze.linalg.{ DenseVector => BDV, Vector => BV, norm => breezeNorm }
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.Logging
import org.apache.spark.annotation.Experimental

class FSIndex private (private var m: Double)
  extends Serializable with Logging {

  /**
   *  Default constructor
   */
  def this() = this(2)

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

  /** Set the value of the fuzziness constant (m). Default: 2. */
  def setM(m: Double): this.type = {
    this.m = m
    this
  }

  /**
   * Format data for the execution
   */
  def formatData(data: RDD[Vector], ithCenters: Array[Array[Vector]]): scala.collection.Map[Int, Double] = {
    // Compute squared norms and cache them.    
    val normsData = data.map(v => breezeNorm(v.toBreeze, 2.0))
    normsData.persist()
    val breezeData = data.map(_.toBreeze).zip(normsData).map {
      case (v, norm) =>
        new BreezeVectorWithNorm(v, norm)
    }
    //Transform prototypes in BreezeVectorWithNorm    
    val normCenters = ithCenters.flatMap(_.map(v => breezeNorm(v.toBreeze, 2.0)))
    println("Norm Centers")
    normCenters.foreach(println)
    //Compute the Vector's norm
    val norm = ithCenters.map(_.map(v => breezeNorm(v.toBreeze, 2.0)))
    //Associate vector with norm
    var i = -1
    val breezeCenters = ithCenters.
      map { v =>
        i += 1
        v.map(x => x.toBreeze).zip(norm(i)).map {
          case (v, norm) =>
            new BreezeVectorWithNorm(v, norm)
        }
      }

    val model = indexes(breezeData, breezeCenters)
    normsData.unpersist()
    model
  }

  /**
   * Compute the Fukuyama-Sugeno index for each FCM execution
   */

  def indexes(data: RDD[BreezeVectorWithNorm],
              ithCenters: Array[Array[BreezeVectorWithNorm]]): scala.collection.Map[Int, Double] = {

    val initStartTime = System.nanoTime()
    val sc = data.sparkContext
    val nExec = ithCenters.length //Number of FCM Executions

    val numData = data.count().toDouble
    val sumData = data.map(a => a.vector).reduce((a, b) => a + b)
    val grandMedia = new BreezeVectorWithNorm(sumData / numData)

    //Compute ||c(j) - grandMedia(c)|| for ech FCM execution and for each c(j) of course
    val centersDist = for (i <- 0 until nExec) yield {
      val centroids = ithCenters(i) //Centers' vector at z-th iteration
      val c = centroids.length //Number of centers at z-th iteration      
      val distances = Array.fill[Double](c)(0)
      for (j <- 0 until c) {
        distances(j) = (KMeans.fastSquaredDistance(centroids(j), grandMedia) + EPSILON)
      }
      (i, distances)
    }
    //Broadcast variables
    val broadcastCenters = sc.broadcast(ithCenters)
    val broadcastCentersDist = sc.broadcast(centersDist)
    val broadcastExecN = sc.broadcast(nExec)
    val broadcastCorrection = sc.broadcast(EPSILON)

    val indices = data.mapPartitions { points =>
      val localFS = Array.fill[Double](broadcastExecN.value)(0)

      points.foreach { point =>
        for (i <- 0 until broadcastExecN.value) { // i will be the index that refers to the number of the FCM's execution
          val centroids = broadcastCenters.value(i).clone() //Centers' vector at z-th iteration
          val c = centroids.length //Number of centers at z-th iteration         
          val singleDist = Array.fill[Double](c)(0)
          val numDist = Array.fill[Double](c)(0)
          var denom = 0.0

          for (j <- 0 until c) {
            //singleDist(j) = KMeans.fastSquaredDistance(point, broadcastCenters.value(i)(j))
            singleDist(j) = KMeans.fastSquaredDistance(point, centroids(j))
            if (singleDist(j) == 0) { singleDist(j) += broadcastCorrection.value }
            numDist(j) = math.pow(singleDist(j), (2 / (m - 1)))
            denom += (1 / numDist(j))
          }

          for (j <- 0 until c) {
            val u = math.pow((numDist(j) * denom), -m) //uij^m            
            localFS(i) += (u * (singleDist(j) - broadcastCentersDist.value(i)._2(j))) // singleDist(j))
          }
        }
      } //foreach end

      val validityIndexLoc = for (i <- 0 until broadcastExecN.value) yield {
        (i, localFS(i))

      }
      validityIndexLoc.iterator

    }.reduceByKey((x, y) => x + y).collectAsMap()

    broadcastCenters.destroy(true)
    broadcastCentersDist.destroy(true)
    broadcastExecN.destroy(true)
    broadcastCorrection.destroy(true)

    indices
  }

}

/**
 * Top-level methods for calling Fukuyama - Sugeno Validity Index Calculation
 */
object FSIndex {

  /**
   * Trains a Fuzzy C-means model using the given set of parameters.
   *
   * @param data training points stored as `RDD[Array[Double]]`
   * @param ithCenters Array of Vector Centers. (We will obtain one vector of centers for each FCM execution)
   * @param m fuzzyfication constant
   */
  def compute(
    data: RDD[Vector],
    ithCenters: Array[Array[Vector]],
    m: Double): scala.collection.Map[Int, Double] = {
    new FSIndex().setM(m)
      .formatData(data, ithCenters)
  }

}
