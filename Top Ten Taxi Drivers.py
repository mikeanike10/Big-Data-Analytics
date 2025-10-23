from __future__ import print_function

import os
import sys
import requests
from operator import add

from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

from pyspark.sql.types import *
from pyspark.sql import functions as func
from pyspark.sql.functions import *

import numpy as np


#Exception Handling and removing wrong datalines
def isfloat(value):
    try:
        float(value)
        return True
 
    except:
         return False

#Function - Cleaning
# checking if the trip distance and fare amount is a float number
# checking if the trip duration is more than a minute, trip distance is more than 0 miles, 
# fare amount and total amount are more than 0 dollars
def correctRows(p):
    if(len(p)==17):
        if(isfloat(p[5]) and isfloat(p[11])):
            if(float(p[4])> 60 and float(p[5])>0 and float(p[11])> 0 and float(p[16])> 0):
                return p

#Main
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: main_task1 <file> <output> ", file=sys.stderr)
        exit(-1)
    
    sc = SparkContext(appName="Assignment-1")
    spark = SparkSession(sc)
    rdd=sc.textFile("taxi-data-sorted-small.csv")
    rdd = spark.read.options(header='false', inferSchema='true',  sep =",")\
        .csv(sys.argv[1])\
            .rdd.map(tuple)\
                .filter(correctRows)\
                    .cache()

    #Task 1 list top 20 drivers by trip amount
    def addToList(x, y):
        x.append(y)
        return x

    def myCombiner(x,y):
        x.extend(y)
        return x
    
    rddtask1=rdd.map(lambda x:(x[0],x[1]))\
        .aggregateByKey(list(),addToList,myCombiner)\
            .map(lambda x:(x[0],len(np.unique(x[1])))) \
                           .top(10,lambda x:x[1])
    rddtask1=sc.parallelize(rddtask1)
    rddtask1.coalesce(1).saveAsTextFile(sys.argv[2])

    #Task 2 Top ten drivers by money made
    rddtask2=rdd.map(lambda x:(x[1],x[4],x[16])) \
        .map(lambda x:(x[0],x[1]/60,x[2])) \
            .map(lambda x:(x[0],x[2]/x[1])) \
                .reduceByKey(lambda x,y:(x+y)/2) \
                    .map(lambda x:(x[0],x[1]))\
                        .top(10, lambda x:x[1])

    #savings output to argument
    rddtask2=sc.parallelize(rddtask2)
    rddtask2.coalesce(1).saveAsTextFile(sys.argv[3])



    sc.stop()
