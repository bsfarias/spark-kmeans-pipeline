from pyspark.ml import Pipeline
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.sql.functions import asc
from pyspark.ml.feature import StandardScaler
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans

#create spark session
spark = SparkSession.builder.appName("spark-kmeans").getOrCreate()

#read raw data
df_raw = spark.read \
         .option("delimiter",";") \
         .option("header",True) \
         .option("inferSchema", "true") \
         .csv("data/*")
df_raw.createOrReplaceTempView("tmp")

#train dataset
training = spark.sql("""SELECT 
                        classificacao_acidente,
                        cast(replace(latitude,',','.') as double)latitude,
                        cast(replace(longitude,',','.')  as double)longitude,
                        cast(mortos as int)mortos,
                        cast( feridos_leves as int)feridos_leves,
                        cast(feridos_graves as int)feridos_graves,
                        cast(feridos as int)feridos,
                        cast(ilesos as int)ilesos,
                        cast(veiculos as int)veiculos
                        FROM tmp""")

#pipeline
vecAssembler = VectorAssembler(inputCols=["latitude","longitude","mortos","feridos_leves","feridos_graves","feridos","ilesos","veiculos"], outputCol="features")
scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
#scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",withStd=True, withMean=False)
kmeans = KMeans().setK(5).setSeed(1).setMaxIter(300).setFeaturesCol("scaledFeatures").setPredictionCol("prediction")
pipeline = Pipeline(stages=[vecAssembler, scaler, kmeans])

#fit
kMeansPredictionModel = pipeline.fit(training)
predictionResult = kMeansPredictionModel.transform(training)

#silhouette
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictionResult)
print("Silhouette with squared euclidean distance = " + str(silhouette)) 

#result
predictionResult.groupBy("prediction","classificacao_acidente").count().show()