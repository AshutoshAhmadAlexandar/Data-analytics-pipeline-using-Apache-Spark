# Import packages
import time
import pyspark
import os
import csv
from numpy import array
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext, SparkConf
from pyspark import SparkSession
from pyspark import SQLContext
# Import packages
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, OneHotEncoder, VectorAssembler, IndexToString
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import *

# Creating Spark environment
#os.environ["HADOOP_USER_NAME"] = "hdfs"
#os.environ["PYTHON_VERSION"] = "3.5.2"
#conf = pyspark.SparkConf()
#sc = pyspark.SparkContext(conf=conf)
#conf.getAll()

sc = SparkContext()
sqlCont = SQLContext(sc)
 
# Creatingt Spark SQL environment
#from pyspark.sql import SparkSession, HiveContext
#SparkContext.setSystemProperty("hive.metastore.uris", "thrift://nn1:9083")
#spark = SparkSession.builder.enableHiveSupport().getOrCreate()
 
# spark is an existing SparkSession
#train = spark.read.csv("file:/home/hadoop/Downloads/Spark/Part1/train.csv", header = True)
# Displays the content of the DataFrame to stdout
#train.show(10)

train = sqlCont.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('data.csv')
# Displays the content of the DataFrame to stdout
train.show(10)
 
# String to float on some columns of the dataset : creates a new dataset
train = train.select(col("Survived"),col("Sex"),col("Embarked"),col("Pclass").cast("float"),col("Age").cast("float"),col("SibSp").cast("float"),col("Fare").cast("float"))
 
# dropping null values
train = train.dropna()
 
# Spliting in train and test set. Beware : It sorts the dataset
(traindf, testdf) = train.randomSplit([0.7,0.3])

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
genderIndexer = StringIndexer(inputCol="Sex", outputCol="indexedSex")
embarkIndexer = StringIndexer(inputCol="Embarked", outputCol="indexedEmbarked")
 
surviveIndexer = StringIndexer(inputCol="Survived", outputCol="indexedSurvived")
 
# One Hot Encoder on indexed features
genderEncoder = OneHotEncoder(inputCol="indexedSex", outputCol="sexVec")
embarkEncoder = OneHotEncoder(inputCol="indexedEmbarked", outputCol="embarkedVec")
 
# Create the vector structured data (label,features(vector))
assembler = VectorAssembler(inputCols=["Pclass","sexVec","Age","SibSp","Fare","embarkedVec"],outputCol="features")
 
# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedSurvived", featuresCol="features")
 
# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[surviveIndexer, genderIndexer, embarkIndexer, genderEncoder,embarkEncoder, assembler, rf]) # genderIndexer,embarkIndexer,genderEncoder,embarkEncoder,
 
# Train model.  This also runs the indexers.
model = pipeline.fit(traindf)
 
# Predictions
predictions = model.transform(testdf)
 
# Select example rows to display.
predictions.columns 
 
# Select example rows to display.
predictions.select("prediction", "Survived", "features").show(5)
 
# Select (prediction, true label) and compute test error
predictions = predictions.select(col("Survived").cast("Float"),col("prediction"))
evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))
 
rfModel = model.stages[6]
print(rfModel)  # summary only
 
evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g" % accuracy)
 
evaluatorf1 = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="f1")
f1 = evaluatorf1.evaluate(predictions)
print("f1 = %g" % f1)
 
evaluatorwp = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="weightedPrecision")
wp = evaluatorwp.evaluate(predictions)
print("weightedPrecision = %g" % wp)
 
evaluatorwr = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="weightedRecall")
wr = evaluatorwr.evaluate(predictions)
print("weightedRecall = %g" % wr)
 
# close sparkcontext
sc.stop()
