#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:10:14 2023

@author: arthurglass
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
#For example, remove lines if they don’t have 16 values and 
# checking if the trip distance and fare amount is a float number
# checking if the trip duration is more than a minute, trip distance is more than 0.1 miles, 
# fare amount and total amount are more than 0.1 dollars
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
    
    sc = SparkContext(appName="Assignment-3")
    
    rdd = sc.textFile(sys.argv[1])

    #Task 1
    #Your code goes here


    # Results_1 should have m and b parameters from the calculations
    results_1.coalesce(1).saveAsTextFile(sys.argv[2])


    #Task 2
    #Your code goes here


    # print the cost, intercept, and the slope for each iteration

    # Results_2 should have m and b parameters from the gradient Descent Calculations
    results_2.coalesce(1).saveAsTextFile(sys.argv[3])




    #Task 3 
    #Your code goes here


    # print the cost, intercept, the slopes (m1,m2,m3,m4), and learning rate for each iteration

    # Results_3 should have b, m1, m2, m3, and m4 parameters from the gradient Descent Calculations
    results_3.coalesce(1).saveAsTextFile(sys.argv[4])
    

    sc.stop()