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

  def correction: Double = 0.000001 //Correction parameter for euclidean distance computation

  def this() = this(2)

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
    val breezeCenters = ithCenters.map(v => v.map(x => x.toBreeze).zip(normCenters).map {
      case (v, norm) =>
        new BreezeVectorWithNorm(v, norm)
    })

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
    val nExec = ithCenters.length //Number of FCM Executions
    val sc = data.sparkContext
    val broadcastVar = sc.broadcast(ithCenters,nExec)
    
    val numData = data.count().toDouble 
    
    val sumData = data.map(a => a.vector).reduce((a, b) => a + b)
    val grandMedia = new BreezeVectorWithNorm(sumData / numData)

      //Compute ||c(j) - grandMedia(c)|| for ech FCM execution and for each c(j) of course
    val centersDist = for (i <- 0 until nExec) yield {
      val centroids = ithCenters(i) //Centers' vector at z-th iteration
      val c = centroids.length //Number of centers at z-th iteration      
      val distances = Array.fill[Double](c)(0)
      for (j <- 0 until c) {
        distances(j) = (KMeans.fastSquaredDistance(centroids(j), grandMedia) + correction)
      }      
      (i, distances)      
    }
    
    
    val localFS = Array.fill[Double](nExec)(0)

    val indexes = data.mapPartitions { points =>

      
      
      points.foreach { point =>

      for (i <- 0 until nExec) {                   // z will be the index that refers to the number of the FCM's execution

          val centroids = ithCenters(i)           //Centers' vector at z-th iteration
          val c = centroids.length               //Number of centers at z-th iteration          
          val singleDist = Array.fill[Double](c)(0)
          val numDist = Array.fill[Double](c)(0)
          var denom = 0.0

          for (j <- 0 until c) {
            singleDist(j) = (KMeans.fastSquaredDistance(point, centroids(j)) + correction)
            numDist(j) = math.pow(singleDist(j), (2 / (m - 1)))            
            denom += (1 / numDist(j))
          }

          for (j <- 0 until c) {
            val u = math.pow((numDist(j) * denom), -m) //uij^m
            localFS(i) +=  (u * centersDist(i)._2(j))
            //* singleDist(j)) 
          }
        }
      } //foreach end

      val validityIndexContr = for (i <- 0 until nExec) yield {
        (i, localFS(i))
      }
      
      validityIndexContr.iterator
    }.reduceByKey((x, y) => (x + y)).collectAsMap()

    indexes
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
