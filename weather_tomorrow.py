import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import datetime

from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('colour prediction').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
assert spark.version >= '2.4' # make sure we have Spark 2.4+

from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
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


def test_model(model_file):
    model = PipelineModel.load(model_file)
    inputs = [('sfu',datetime.date(2023, 11, 17), 49.2771, -122.9146, 330.0, 12.0), ('sfu',datetime.date(2023, 11, 18), 49.2771, -122.9146, 330.0, 12.0)]
    test = spark.createDataFrame(inputs, schema = tmax_schema)
    predictions = model.transform(test)
    result = predictions.select('prediction').collect()
    result = result[0]['prediction']
    print('Predicted tmax tomorrow:', result)

if __name__ == '__main__':
    model_file = sys.argv[1]
    test_model(model_file)

