import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('colour prediction').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
assert spark.version >= '2.4' # make sure we have Spark 2.4+

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, SQLTransformer
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator





tmax_schema = types.StructType([
    types.StructField('station', types.StringType()),
    types.StructField('date', types.DateType()),
    types.StructField('latitude', types.FloatType()),
    types.StructField('longitude', types.FloatType()),
    types.StructField('elevation', types.FloatType()),
    types.StructField('tmax', types.FloatType()),
])

def main(inputs, model_file):
    data = spark.read.csv(inputs, schema = tmax_schema)
    train, validation = data.randomSplit([0.75, 0.25])
    train = train.cache()
    validation = validation.cache()

    sqlTrans = SQLTransformer(statement = """SELECT today.station as station,today.latitude as latitude,today.longitude as longitude,today.elevation as elevation, dayofyear( today.date ) AS day, today.tmax, yesterday.tmax AS yesterday_tmax FROM __THIS__ as today INNER JOIN __THIS__ as yesterday ON date_sub(today.date, 1) = yesterday.date AND today.station = yesterday.station""")
    weather_assembler = VectorAssembler(inputCols=["latitude", "longitude", "elevation", "day", "yesterday_tmax"], outputCol = "features")
    estimator = GBTRegressor(featuresCol = "features", maxIter = 10, labelCol = 'tmax')
    word_indexer = StringIndexer(inputCol = "station", outputCol = "label")
    pipeline = Pipeline(stages=[sqlTrans, weather_assembler,word_indexer, estimator])
    evaluator = RegressionEvaluator()

    model = pipeline.fit(train)
    predictions = model.transform(validation)
    r2 = evaluator.evaluate(predictions,{evaluator.metricName: "r2"})
    rmse = evaluator.evaluate(predictions,{evaluator.metricName: "rmse"})
    model.write().overwrite().save(model_file)

if __name__ == '__main__':
    inputs = sys.argv[1]
    model_file = sys.argv[2]
    main(inputs,model_file)
