// Import the necessary packages
import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.expressions.Window
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.functions.rand
import org.apache.spark.sql.functions._

object smoteClass{
  def KNNCalculation(
    dataFinal:org.apache.spark.sql.DataFrame,
    feature:String,
    reqrows:Int,
    BucketLength:Int,
    NumHashTables:Int):org.apache.spark.sql.DataFrame = {
    val b1 = dataFinal.withColumn("index", row_number().over(Window.partitionBy("label").orderBy("label")))
    val brp = new BucketedRandomProjectionLSH().setBucketLength(BucketLength).setNumHashTables(NumHashTables).setInputCol(feature).setOutputCol("values")
    val model = brp.fit(b1)
    val transformedA = model.transform(b1)
    val transformedB = model.transform(b1)
    val b2 = model.approxSimilarityJoin(transformedA, transformedB, 2000000000.0)
    val b3 = b2.selectExpr("datasetA.index as id1",
        "datasetA.feature as k1",
        "datasetB.index as id2",
        "datasetB.feature as k2",
        "distCol").filter("distCol>0.0").orderBy("id1", "distCol").dropDuplicates().limit(reqrows)
    return b3
  }

  def smoteCalc(key1: org.apache.spark.ml.linalg.Vector, key2: org.apache.spark.ml.linalg.Vector)={
    val resArray = Array(key1, key2)
    val res = key1.toArray.zip(key2.toArray.zip(key1.toArray).map(x => x._1 - x._2).map(_*0.2)).map(x => x._1 + x._2)
    resArray :+ org.apache.spark.ml.linalg.Vectors.dense(res)}

  def Smote(
    inputFrame:org.apache.spark.sql.DataFrame,
    feature:String,
    label:String,
    minorityclass:String,
    percentOver:Int,
    BucketLength:Int,
    NumHashTables:Int):org.apache.spark.sql.DataFrame = {
    val rowCount = inputFrame.count
    val reqrows = (rowCount * (percentOver/100)).toInt
    val md = udf(smoteCalc _)
    val b1 = KNNCalculation(inputFrame, feature, reqrows, BucketLength, NumHashTables)
    val b2 = b1.withColumn("ndtata", md($"k1", $"k2")).select("ndtata")
    val b3 = b2.withColumn("AllFeatures", explode($"ndtata")).select("AllFeatures").dropDuplicates.withColumn("label", lit(minorityclass))
    return b3
  }
}

val dataInput = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/data/sam.csv").na.drop()
val TFN = "Y"
val columnNames = dataInput.columns.filter(_ != TFN)
val assembler1 = new VectorAssembler().setInputCols(columnNames).setOutputCol("feature")
val assembled1 = assembler1.transform(dataInput)
val vectorized = assembled1.select("feature",TFN).withColumn("label",col(TFN)).drop(TFN)
val newData = Smote(vectorized, "feature", "label", "1", 200, 10, 30)
