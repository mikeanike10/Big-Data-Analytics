#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 11:16:48 2023

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
                    if(float(p[11])<=15):
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

    #start
    rdd3=rdd.map(lambda x:(np.array([x[4],x[5],x[11],x[15]]),x[16])).cache()
    theta=np.zeros(4)
    b=0
    learningRate=0.00000001
    numIterations=50
    oldcost=0
    n=rdd3.count()
    
    for i in range(numIterations):
        #grandient x[0],cost x[1],intercept x[2]
        gradientCost=rdd3.map(lambda x:(x[0],(x[1]-(np.dot(x[0],theta)+b))))\
            .map(lambda x:(x[0]*x[1],x[1]**2,-x[1]))\
                .reduce(lambda x,y:(x[0]+y[0],x[1]+y[1],x[2]+y[2]))
        newcost=gradientCost[1]/(2*n)
        print("cost",newcost,"theta",theta,"intercept",b)
        gradient=(-1/n)*gradientCost[0]
        theta=theta-learningRate*gradient
        b=b-learningRate*(2/n)*gradientCost[2]
        if (newcost>oldcost):
            learningRate=learningRate*1.05
        elif(newcost<oldcost):
            learningRate=learningRate*0.95
            
        oldcost=newcost
    


    # print the cost, intercept, the slopes (m1,m2,m3,m4), and learning rate for each iteration

    # Results_3 should have b, m1, m2, m3, and m4 parameters from the gradient Descent Calculations
    #results_3.coalesce(1).saveAsTextFile(sys.argv[4])
    sc.stop()
                
    
        

                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
            
