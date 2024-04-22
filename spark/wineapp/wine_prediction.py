import os.path
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics

# Set global variables
MODEL_PATH = '/home/ec2-user/spark/wineapp/trainingmodel.model'
VAL_DATA_PATH = '/home/ec2-user/spark/wineapp/ValidationDataset.csv'
TEST_DATA_PATH = '/home/ec2-user/spark/wineapp/TestDataset.csv'

def predict():
    # Start Spark session
    spark = SparkSession.builder.appName("Wine Quality Prediction").getOrCreate()
    path = ""

    # Determine if TestDataset of ValidationDataset
    if os.path.isfile(TEST_DATA_PATH):
        path = TEST_DATA_PATH
    else:
        path = VAL_DATA_PATH

    # Open and show first 5 datapoints
    val_df = spark.read.csv(path, sep = ';', header = True, inferSchema = True)
    val_df.show(5)

    # Format dataset
    feature_columns = val_df.columns[:-1]
    qual = val_df.columns[-1]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

    # Transform data based on format
    val_df = assembler.transform(val_df)
    final_val_df = val_df.select("features", qual)

    # Load previously trained model
    lr_model = LogisticRegressionModel.load(MODEL_PATH)

    # Use model for predictions
    predictions = lr_model.transform(final_val_df)
    predictions = predictions.withColumn(qual, col(qual).cast("double"))

    # Change data to RDD
    predictionAndLabels = predictions.select("prediction", qual).rdd
    metrics = MulticlassMetrics(predictionAndLabels)

    # Print models F-Score and dataset path
    print("F-Score: ", metrics.weightedFMeasure())
    print(path)

if __name__ == "__main__":

    predict()
