/**
  * Created by qinjianyang on 2017/5/23.
  */
object TFIDF {
  def main(args: Array[String]): Unit = {
    println("feature extract")
    import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
    import org.apache.spark.sql.SparkSession

    val spark = SparkSession
      .builder()
      .appName("spark feature extract")
      .config("spark.master","local")
      .getOrCreate()

    val sentenceData = spark.createDataFrame(Seq(
      (0.0, "Hi I heard about spark"),
      (0.0, "I wish java could use case classes"),
      (1.0, "Logistic regression models are neat")
    )).toDF("label", "sentence")

    sentenceData.select("label", "sentence").show()

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)
    wordsData.select("label", "words").show()

    val hashingTF = new HashingTF()
      .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)

    val featureizeData = hashingTF.transform(wordsData)
    featureizeData.select("label","rawFeatures").collect().foreach(println)

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")

    val idfModel = idf.fit(featureizeData)

    val rescaledData = idfModel.transform(featureizeData)

    rescaledData.select("label","features").collect().foreach(println)
  }
}

object Word2Vec {
  def main(args: Array[String]): Unit = {
    println("word2vec")
    import org.apache.spark.ml.feature.Word2Vec
    import org.apache.spark.ml.linalg.Vector
    import org.apache.spark.sql.Row
    import org.apache.spark.sql.SparkSession
    import org.apache.spark.SparkConf
    import org.apache.spark.SparkContext

    val spark = SparkSession
      .builder()
      .appName("Word2Vec test")
      .config("spark.master","local")
      .getOrCreate()

    /*val conf = new SparkConf().setAppName("TfIdfTest")
    val spark = new SparkContext(conf)*/

    val documentDF = spark.createDataFrame(Seq(
      "Hi I heard about Spark".split(" "),
      "I wish Java could use case classes".split(" "),
      "Logistic regression models are neat".split(" ")
    ).map(Tuple1.apply)).toDF("text")

    val word2Vec = new Word2Vec()
      .setInputCol("text")
      .setOutputCol("result")
      .setVectorSize(3)
      .setMinCount(0)

    val model = word2Vec.fit(documentDF)
    model.getVectors.collect().foreach(println)

    val result = model.transform(documentDF)
    result.collect().foreach{
      case Row(text: Seq[_], features: Vector) =>
        println(s"Text:[${text.mkString(", ")}] => \nVector: $features\n")
    }
  }
}

object CountVectorizer {
  import org.apache.spark.sql.SparkSession
  import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("CountVectorizer")
      .config("spark.master", "local")
      .getOrCreate()

    val df = spark.createDataFrame(Seq(
      (1, Array("a", "b", "c", "d")),
      (2, Array("a" ,"b", "c", "a" ,"b"))
    )).toDF("id","words")

    val cvModel: CountVectorizerModel = new CountVectorizer()
      .setInputCol("words")
      .setVocabSize(3)
      .setOutputCol("features")
      .setMinDF(2)
      .fit(df)

    val cvm = new CountVectorizerModel(Array("a", "b", "c", "d"))
      .setInputCol("words")
      .setOutputCol("features")

    cvm.transform(df).show()
    cvm.transform(df).select("id","features").collect().foreach(println)

    cvModel.transform(df).show()
    cvModel.transform(df).select("id", "features").collect().foreach(println)
  }
}