package org.apache.spark.mllib.clustering

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import breeze.linalg.{ DenseVector => BDV, Vector => BV, norm => breezeNorm }
import org.apache.spark.mllib.util.MLUtils




class FSIndex (val ithCenters: Array[Array[Vector]], m: Double) extends Serializable {
  
    def nExec: Int = ithCenters.length      //Number of FCM Executions

    
    /**
     * Format data for the execution
     */
    def formatData(data: RDD[Vector], ithCenters: Array[Array[Vector]]): Array[Double] = {
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
    
    def indexes(data :RDD[BreezeVectorWithNorm],
        ithCenters:Array[Array[BreezeVectorWithNorm]]): Array[Double] = {
     val fs = Array.fill[Double](nExec)(0)
      
      
     
     
     fs
    } 
     

    
    
}