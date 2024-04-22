FROM python:3.8-slim

# environment variables
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# install Java and PySpark dependencies
RUN apt-get update && apt-get install -y openjdk-17-jre-headless && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Spark variables
ENV SPARK_VERSION=3.5.1
ENV HADOOP_VERSION=3
ENV SPARK_PACKAGE=spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}
ENV SPARK_HOME=/usr/local/spark
ENV PATH=$PATH:${SPARK_HOME}/bin

# download Spark
RUN wget -qO- https://downloads.apache.org/spark/spark-${SPARK_VERSION}/${SPARK_PACKAGE}.tgz | tar xz -C /usr/local/
ENV SPARK_HOME=/usr/local/${SPARK_PACKAGE}

# install Python packages
RUN pip install pyspark
RUN pip install numpy

# copy code to app
COPY . /wineapp
WORKDIR /wineapp

# run application on start
CMD ["spark-submit", "wine_prediction.py"]
