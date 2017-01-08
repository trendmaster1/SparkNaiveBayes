package Classifiers

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.classification.NaiveBayesModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.sql.SparkSession

// Naive Bayes classification on Spark

class SparkNaiveBayes(sc: SparkContext) {  
    // training function
    def training(trainingData: RDD[LabeledPoint], modelPath: String) {
        // train model
        val model = NaiveBayes.train(trainingData, lambda = 1.0)	
        
        // save model
        model.save(sc, modelPath);
    }
    
    // predict function
    def predict(inputData: RDD[LabeledPoint], modelPath: String) = {
        // load model
        val model = NaiveBayesModel.load(sc, modelPath)
       
        // predict
        val predictionAndLabel = inputData.map(p => (model.predict(p.features), p.label))
    
        //compute accuracy
        val accuracy = 1.0 * predictionAndLabel.filter(label => label._1 == label._2).count()	
        
        // return result
        (predictionAndLabel, accuracy)
     }   
}

 
object TestSparkNaiveBayes {
     // set up spark context
    val conf = new SparkConf()   
      .setMaster("local")        
      .setAppName("SparkNativeBayes") 
    val sc = new SparkContext(conf)
 
    val spark = SparkSession
      .builder()
      .config("spark.sql.warehouse.dir", "file:///c:/tmp/spark-warehouse")
      .getOrCreate()   
      
    // define classifier
    val classifier = new SparkNaiveBayes(sc);

    // main function
    def main(args: Array[String]) {
    
    // load data
    val data = sc.textFile("data.txt")
    
    // parse data
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }

    val splits = parsedData.randomSplit(Array(0.7, 0.3), seed = 11L)		
    
    // set up training data
    val trainingData = splits(0)	
    
    //set up test data
    val testData = splits(1)						
    
    // train model
    classifier.training(trainingData, "naivebayesmodel")	
    
    // predict
    val (predictionAndLabel, accuracy) = classifier.predict(testData, "naivebayesmodel")
    
    println(accuracy)
  }
}