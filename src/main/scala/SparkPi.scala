
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.metrics._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{Row, SparkSession}

/**
- @author zhengsd
  */
object SparkPi {
  def main(args: Array[String]) {
    println("hello world")
    /*val conf = new SparkConf()
    val sc = new SparkContext(conf)
    val text = sc.textFile("file:///usr/local/README.md")
    val result = text.flatMap(_.split(' ')).map((_, 1)).reduceByKey(_ + _).collect()
    result.foreach(println)*/


    val spark = SparkSession
      .builder()
      .appName("Spark SQL basic example")
      .config("spark.master","local")
      .getOrCreate()

    val training = spark.createDataFrame(Seq(
      (1.0, Vectors.dense(0.0, 1.1, 0.1)),
      (0.0, Vectors.dense(2.0, 1.0, -1.0)),
      (0.0, Vectors.dense(2.0, 1.3, 1.0)),
      (1.0, Vectors.dense(0.0, 1.2, -0.5))
    )).toDF("label", "features")

    //创建logisticRegression实例
    val lr = new LogisticRegression()

    println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

    lr.setMaxIter(10)
      .setRegParam(0.01)

    val model1 = lr.fit(training)

    println("Model 1 was fit using parameters: " + model1.parent.explainParams())

    val paramMap = ParamMap(lr.maxIter -> 20)
      .put(lr.maxIter,30)
      .put(lr.regParam -> 0.1, lr.threshold -> 0.55)

    val paramMap2 = ParamMap(lr.probabilityCol -> "myProbability")
      .put(lr.maxIter,20)
    val paramMapCombined = paramMap ++ paramMap2

    val model2 = lr.fit(training, paramMapCombined)
    println("Model 2 was fit using parameters: " + model2.parent.explainParams())

    val test = spark.createDataFrame(Seq(
      (1.0, Vectors.dense(-1.0, 1.5, 1.3)),
      (0.0, Vectors.dense(3.0, 2.0, -1.0)),
      (1.0, Vectors.dense(0.0, 2.2, -1.5))
    )).toDF("label", "features")

    model2.transform(test)
      .select("features", "label", "myProbability", "prediction")
      .collect()
      .foreach{
        case Row(features: Vector, label: Double, prob: Vector, prediction: Double) =>
        println(s"($features,$label) -> prob=$prob,prediction=$prediction")
      }
  }
}

object Pipeline{
  import org.apache.spark.ml.{Pipeline,PipelineModel}
  import org.apache.spark.ml.classification.LogisticRegression
  import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
  import org.apache.spark.ml.linalg.Vector
  import org.apache.spark.sql.Row
  def main(args: Array[String]): Unit = {
    println("pipelines")

    val spark = SparkSession
      .builder()
      .appName("Spark SQL basic example")
      .config("spark.master","local")
      .getOrCreate()

    val training = spark.createDataFrame(Seq(
      (0L, "a b c d e spark", 1.0),
      (1L, "b d", 0.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 0.0)
    )).toDF("id", "text", "label")

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.001)

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, lr))

    val model = pipeline.fit(training)

    model.write.overwrite().save("/tmp/spark/spark-logistic-regression-model")

    pipeline.write.overwrite().save("/tmp/spark/unfit-lr-model")

    val sameModel = PipelineModel.load("/tmp/spark/spark-logistic-regression-model")

    val test = spark.createDataFrame(Seq(
      (4L, "spark i j k"),
      (5L, "l m n"),
      (6L, "spark hadoop spark"),
      (7L, "apache hadoop")
    )).toDF("id","text")

    model.transform(test)
      .select("id", "text", "probability", "prediction")
      .collect()
      .foreach{
        case Row(id:Long, text:String, prob:Vector, prediction:Double) =>
          println(s"($id,$text) --> prob=$prob,prediction=$prediction")
      }
  }
}