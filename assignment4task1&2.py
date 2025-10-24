#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:21:51 2023

@author: arthurglass
"""

from __future__ import print_function

import re
import sys
import numpy as np
from operator import add

from pyspark import SparkContext
numTopWords = 10000
def freqArray (listOfIndices):
	global numTopWords
	returnVal = np.zeros (numTopWords)
	for index in listOfIndices:
		returnVal[index] = returnVal[index] + 1
	mysum = np.sum (returnVal)
	returnVal = np.divide(returnVal, mysum)
	return returnVal
def indexofword(word):
    try:
        print("Index for",word,"is",dictionary.filter(lambda x: x[0]==word).take(1)[0][1])
    except:
        print("index for",word,"is -1")
        
if __name__ == "__main__":
    sc = SparkContext(appName="Assignment-4")
    
    ### Task 1
	### Data Preparation
    corpus = sc.textFile(sys.argv[1],1)
    keyAndText = corpus.map(lambda x :(x[x.index('id="')+4:x.index('" url=')],x[x.index('">')+2:][:-6]))
    regex = re.compile('[^a-zA-Z]')
    
    keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
    
    #word and count 
    allWords=keyAndListOfWords.flatMap(lambda x:[(y,1) for y in x[1]])
    
    #counts all words
    allCounts = allWords.reduceByKey(lambda x, y: x+y)

    #top 10,000 words
    topWords = np.array(allCounts.sortBy(lambda x:x[1],ascending=False).top(numTopWords,lambda x:x[1]))    
    topWordsK = sc.parallelize(range(numTopWords))
    
    #dictionary
    dictionary = topWordsK.map (lambda x : (topWords[x][0], x)).cache()
    
    ### Include the following results in your report:
    indexofword("applicant")
    indexofword("and")
    indexofword("attack")
    indexofword("protein")
    indexofword("car")
    indexofword("in")
    
    ### Task 2
	### Build your learning model
    
    #TF vector
    #word, docID
    allWordsWithDocID = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))
    
    #join allwords with dictionary
    allDictionaryWords = allWordsWithDocID.join(dictionary)
    
    #doc and word dictionary position
    justDocAndPos = allDictionaryWords.map(lambda x:x[1])
    
    #dictionary words in each doc
    allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()
    
    #docs and their term frequencies
    traindata = allDictionaryWordsInEachDoc.map(lambda x: (x[0], freqArray(x[1]))).map(lambda x: (1 if x[0][0:2]=='AU' else 0,x[1])).cache()
    
    
    #weights
    trainsize=traindata.count()
    w1=trainsize/(2*traindata.filter(lambda x:x[0]==1).count())
    w0=trainsize/(2*traindata.filter(lambda x:x[0]==0).count())
    
    
    
    def logisticRegression(traindata=traindata,
                           max_iteration=50,
                           learningRate=0.01,
                           regularization=0.01,
                           mini_batch_size=50000,
                           tolerance=10e-8,
                           optimizer="SGD",
                           beta2=0.999,
                           train_size=1
                           ):
        #initializations
        prevCost=0
        L_cost=[]
        
        parameterSize=len(traindata.take(1)[0][1])
        np.random.seed(0)
        parameterVector=np.zeros(parameterSize)
        momentum=np.zeros(parameterSize)
        prev_mom=np.zeros(parameterSize)
        second_mom=np.array(parameterSize)
        gti=np.zeros(parameterSize)
        epsilon=10e-8
        
        
        for i in range(max_iteration):
            
            beta=parameterVector
            min_batch=traindata.sample(False,mini_batch_size/trainsize,1+i)
                
            #class 1
            res1=min_batch.filter(lambda x:x[0]==1).treeAggregate(
                (np.zeros(parameterSize),0,0),
                lambda x,y:(x[0]\
                            +y[1]*(-y[0]+(np.exp(np.dot(y[1],beta))\
                                          /(1+np.exp(np.dot(y[1],beta))))),\
                                x[1]\
                                    +y[0]*(-(np.dot(y[1],beta)))\
                                        +np.log(1+np.exp(np.dot(y[1],beta))),
                                        x[2]+1),
                lambda x,y:(x[0]+y[0],x[1]+y[1],x[2]+y[2]))
              
            #class 0
            res0=min_batch.filter(lambda x:x[0]==0).treeAggregate(
                (np.zeros(parameterSize),0,0),
                lambda x,y:(x[0]\
                            +y[1]*(-y[0]+(np.exp(np.dot(y[1],beta))\
                                          /(1+np.exp(np.dot(y[1],beta))))),\
                                x[1]\
                                    +y[0]*(-(np.dot(y[1],beta)))\
                                        +np.log(1+np.exp(np.dot(y[1],beta))),
                                        x[2]+1),
                lambda x,y:(x[0]+y[0],x[1]+y[1],x[2]+y[2]))
            
            #sums
            gradients=w0*res0[0]+w1*res1[0]
            sum_cost=w0*res0[1]+w1*res1[1]
            num_samples=res0[2]+res1[2]
            
            #cost
            cost = sum_cost/num_samples+regularization*(np.square(parameterVector).sum())
            
            #gradient derivative 
            gradientDerivative = (1/num_samples)*gradients+(2*regularization*parameterVector)
            
            if optimizer == 'SGD':
                parameterVector=parameterVector-learningRate*gradientDerivative
                
            if optimizer == 'momentum':
                momentum=beta*momentum+learningRate*gradientDerivative
                parameterVector=parameterVector-momentum
                
            if optimizer == 'nesterov':
                parameter_temp = parameterVector - beta * prev_mom
                parameterVector = parameter_temp - learningRate * gradientDerivative
                prev_mom = momentum
                momentum = beta * momentum + learningRate * gradientDerivative
                
            if optimizer == 'Adam':
                momentum = beta * momentum + (1 - beta) * gradientDerivative
                second_mom = beta2 * second_mom + (1 - beta2) * (gradientDerivative**2)
                momentum_ = momentum / (1 - beta**(i + 1))
                second_mom_ = second_mom / (1 - beta2**(i + 1))
                parameterVector = parameterVector - learningRate * momentum_ / (np.sqrt(second_mom_) + epsilon)
                
            if optimizer == 'Adagrad':
                gti += gradientDerivative**2
                adj_grad = gradientDerivative / (np.sqrt(gti)  + epsilon)
                parameterVector = parameterVector - learningRate  * adj_grad
                
            if optimizer == 'RMSprop':
                sq_grad = gradientDerivative**2
                exp_grad = beta * gti / (i + 1) + (1 - beta) * sq_grad
                parameterVector = parameterVector - learningRate / np.sqrt(exp_grad + epsilon) * gradientDerivative
                gti += sq_grad
            
            print("iteration:", i, "cost=",cost)
            
            # Stop if the cost is not descreasing
            if abs(cost - prevCost) < tolerance:
                print("cost - prev_cost: " + str(cost - prevCost))
                break
            prevCost = cost
            L_cost.append(cost)
        
        return parameterVector, L_cost
    
    #parameter_vector_sgd,L_cost=logisticRegression(traindata=traindata,regularization=0)
   # parameter_vecotr_sgd1,L_cost=logisticRegression(traindata=traindata)
    parameter_vector_mom,L_cost=logisticRegression(traindata=traindata,learningRate=0.001,optimizer="momentum")
   # parameter_vector_nes,L_cost=logisticRegression(traindata=traindata, optimizer="nesterov")
   # parameter_vector_adam,L_cost=logisticRegression(traindata=traindata,optimizer="Adam")
   # parameter_vector_ada,L_cost=logisticRegression(traindata=traindata,optimizer="Adagrad")
   # parameter_vector_rms,L_cost=logisticRegression(traindata=traindata,optimizer="RMSprop")
    
    ### Print the top 5 words with the highest coefficients
    dict50=dictionary.collect(50)
    dictionary.take(1)
    len(parameter_vector_mom)  
    
    
    
    sc.stop()
    
    
    
    
    
    
    
    
    
    
    
    
