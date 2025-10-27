#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 17:39:00 2023

@author: Michael Glass
"""

from __future__ import print_function

import re
import sys
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.functions import concat_ws, col, split, translate, date_format
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from array import array
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

if __name__ == "__main__":

    sc = SparkContext(appName="project")
    spark = SparkSession(sc)
    names=["precipProbability","temperature","apparentTemperature","dewPoint","humidity","pressure","windSpeed","windGust","windBearing","cloudCover","uvIndex","visibility","ozone"]
    #importing datasets
    #flight
    dfFlight = spark.read.option("delimite",",").option("header",True).csv("Flight_on_time_HIX.csv").drop("Departure_Taxi","Departure_WheelsOff","Scheduled_Arrival_Time","Actual_Arrival_Time","Arrival_Taxi","Arrival_WheelsOn","Delay_Reason","Flight_Distance","Scheduled_Departure_Time","Actual_Departure_Time","Destination_Airport","Arrival_Delay_Minutes","Airline","Flight_Number","Plane_ID")\
        .filter("Origin_Airport=='HIX'")\
            .withColumn("Departure_Delay_Minutes",col("Departure_Delay_Minutes").cast("float"))\
                .select(concat_ws(" ","Origin_Airport","FlightDate").alias("FID"),"Departure_Delay_Minutes")
                
    

    #weather dictionary 
    dfDict = spark.read.option("delimite",",").option("header",True).csv("Airport_weather.csv").drop("summary","precipType","precipAccumulation","time2","_c0").na.replace("Highland","HIX")\
        .na.drop(how="any")\
            .withColumn("precipIntensity", col("precipIntensity").cast("float"))\
            .withColumn("precipProbability", col("precipProbability").cast("float"))\
            .withColumn("temperature", col("temperature").cast("float"))\
            .withColumn("apparentTemperature", col("apparentTemperature").cast("float"))\
            .withColumn("dewPoint", col("dewPoint").cast("float"))\
            .withColumn("humidity", col("humidity").cast("float"))\
            .withColumn("pressure", col("pressure").cast("float"))\
            .withColumn("windSpeed", col("windSpeed").cast("float"))\
            .withColumn("windGust", col("windGust").cast("float"))\
            .withColumn("windBearing", col("windBearing").cast("float"))\
            .withColumn("cloudCover", col("cloudCover").cast("float"))\
            .withColumn("uvIndex", col("uvIndex").cast("float"))\
            .withColumn("visibility", col("visibility").cast("float"))\
            .withColumn("ozone", col("ozone").cast("float"))\
                .select(concat_ws(" ","airport",date_format(split("time"," ").getItem(0),"MM/dd/yyyy")).alias("DID"),"precipProbability","temperature","apparentTemperature","dewPoint","humidity","pressure","windSpeed","windGust","windBearing","cloudCover","uvIndex","visibility","ozone")\
                    .distinct()
    
    
    df=dfDict.join(dfFlight,dfFlight.FID==dfDict.DID).select("precipProbability","temperature","apparentTemperature","dewPoint","humidity","pressure","windSpeed","windGust","windBearing","cloudCover","uvIndex","visibility","ozone","Departure_Delay_Minutes")
    
    #train and test split
    split=df.randomSplit([0.66,0.34])
    
    #pipeline
    assembler=VectorAssembler(inputCols=["precipProbability","temperature","apparentTemperature","dewPoint","humidity","pressure","windSpeed","windGust","windBearing","cloudCover","uvIndex","visibility","ozone","Departure_Delay_Minutes"],
                              outputCol="features")
    train=assembler.transform(split[0])
    test=assembler.transform(split[1])
    #k-means grouping
    kmeans = KMeans().setK(3).setSeed(1)
    model = kmeans.fit(train)
    pred=model.transform(test)
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(pred)
    print("Silhouette with squared euclidean distance = " + str(silhouette))
    
    # Shows the result.
    centers = model.clusterCenters()
    print("Cluster Centers: ")
    for center in centers:
        print(center)
        
    #linear regression
    lr = LinearRegression(featuresCol="features",labelCol="Departure_Delay_Minutes",maxIter=10, regParam=0.3, elasticNetParam=0.8)
    lrModel = lr.fit(train)

    # Print the coefficients and intercept for linear regression
    print("Coefficients: %s" % str(lrModel.coefficients))
    print("Intercept: %s" % str(lrModel.intercept))
    
    # Summarize the model over the training set and print out some metrics
    trainingSummary = lrModel.summary
    print("numIterations: %d" % trainingSummary.totalIterations)
    print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
    trainingSummary.residuals.show()
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
      
    
