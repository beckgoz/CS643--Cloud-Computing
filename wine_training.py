import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics

# Set global variables
S3_TRAIN_DATA_SOURCE = 's3://cs643-wine-prediction/TrainingDataset.csv'
S3_TEST_DATA_SOURCE = 's3://cs643-wine-prediction/ValidationDataset.csv'
S3_OUTPUT_PATH = 's3://cs643-wine-prediction/results'


def build_training_model():
    # Start Spark session
    spark = SparkSession.builder.appName("Wine Quality Test").getOrCreate()

    # Open training dataset
    wine_df = spark.read.csv(S3_TRAIN_DATA_SOURCE, sep = ';', header = True, inferSchema = True)
    wine_df.show(5)

    # Format dataset
    feature_columns = wine_df.columns[:-1]
    qual = wine_df.columns[-1]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

    # Transform and show data based on format
    wine_df = assembler.transform(wine_df)
    final_wine_df = wine_df.select("features", qual)
    final_wine_df.show(5)

    # Train logistic regression model
    lr = LogisticRegression(featuresCol="features", labelCol=qual)
    lr_model = lr.fit(final_wine_df)

    # Save model to S3 bucket
    lr_model.write().overwrite().save(S3_OUTPUT_PATH + '/trainingmodel.model')



if __name__ == "__main__":

    build_training_model()