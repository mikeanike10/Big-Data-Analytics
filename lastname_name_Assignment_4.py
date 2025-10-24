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


if __name__ == "__main__":

	sc = SparkContext(appName="Assignment-4")

	### Task 1
	### Data Preparation
	corpus = sc.textFile(sys.argv[1], 1)
	keyAndText = corpus.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
	regex = re.compile('[^a-zA-Z]')

	keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))



	### Include the following results in your report:
	print("Index for 'applicant' is",dictionary.filter(lambda x: x[0]=='applicant').take(1)[0][1])
	print("Index for 'and' is",dictionary.filter(lambda x: x[0]=='and').take(1)[0][1])
	print("Index for 'attack' is",dictionary.filter(lambda x: x[0]=='attack').take(1)[0][1])
	print("Index for 'protein' is",dictionary.filter(lambda x: x[0]=='protein').take(1)[0][1])
	print("Index for 'car' is",dictionary.filter(lambda x: x[0]=='car').take(1)[0][1])
	print("Index for 'in' is",dictionary.filter(lambda x: x[0]=='in').take(1)[0][1])




	### Task 2
	### Build your learning model



	### Print the top 5 words with the highest coefficients



	### Task 3
	### Use your model to predict the category of each document
	
	print("\nPerformance Metrics: Logisitic Regression with Gradient Descent")
    print("Precision:", ???)
    print("Recall:", ???)
    print("F1:", ???)
    print("Confusion Matrix:", ???)


	sc.stop()