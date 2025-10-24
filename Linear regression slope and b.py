#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:12:39 2023

@author: Michael Glass
"""

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
# checking if the trip duration is more than a minute, trip distance is more than 0.1 miles, 
# fare amount and total amount are more than 0.1 dollars
def correctRows(p):
    if(len(p)==17):
        if(isfloat(p[5]) and isfloat(p[11])):
            if(float(p[4])> 60 and float(p[5])>0 and float(p[11])> 0 and float(p[16])> 0):
                if(float(p[15])<600 or float(p[15]>=1)):
                    return p

#Main
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: main_task1 <file> <output> ", file=sys.stderr)
        exit(-1)
    
    sc = SparkContext(appName="Assignment-3")
    spark = SparkSession(sc)
    
    rdd=spark.read.format('csv').options(header='false', inferSchema='true',  sep =",").load(sys.argv[1])\
        .rdd.map(tuple)\
            .filter(correctRows)\
                .cache() 
    
    
    #Task 1
    #Your code goes here
    rdd1=rdd.map(lambda x:(x[5],x[11])).cache()

    #rdd for x*y sum
    sumxy=rdd1.map(lambda x:x[0]*x[1]).reduce(lambda x,y:x+y)
    
    #rdd for x sum
    sumx=rdd1.map(lambda x:x[0]).reduce(lambda x,y:x+y)
    
    #rdd for x^2 sum
    sumxsquared=rdd1.map(lambda x:x[0]**2).reduce(lambda x,y:x+y)
 
    #rdd for y sum
    sumy=rdd1.map(lambda x:x[1]).reduce(lambda x,y:x+y)
    
    #length
    n=rdd1.count() 

    #slope
    m=(n*sumxy-sumx*sumy)/(n*sumxsquared-sumx**2)
    
    #intercept
    b=(sumxsquared*sumy-sumx*sumxy)/(n*sumxsquared-sumx**2)
    
    print("slope of the data is", m,"and intercept of the data is",b)
    
    # Results_1 should have m and b parameters from the calculations
    #results_1.coalesce(1).saveAsTextFile(sys.argv[2])
    sc.stop()
