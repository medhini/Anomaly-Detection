import cv2
import numpy as np 
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import os, os.path
from skimage import color
import csv
import scipy.io
import scipy.optimize
import time
from operator import add, sub
from time import gmtime, strftime
# from covariance.py import * 

def sigmoid(x):
	return (1 / (1 + np.exp(-x)))

def KLDivergence(rho, rhoCap):
	# print rhoCap
	return np.sum(rho*np.log(rho/rhoCap) + (1 - rho)*np.log((1 - rho)/ (1 - rhoCap)))

#Covariance
def getCovarianceandMean(samples):

	height, width = samples.shape

	#Calculate the mean of each dimension
	i = 0
	j = 0

	means = np.zeros(width, float)
	sums = np.zeros(width, float)

	while(i < width):
		j = 0
		while(j < height):
			sums[i] += samples[j][i]
			j += 1	
		means[i] = sums[i]/height
		i += 1

	#Calculate the variance matrix
	variance = np.zeros(samples.shape, float)

	i = 0
	j = 0

	while(i < width):
		j = 0
		while(j < height):
			variance[j][i] = samples[j][i] - means[i]
			j += 1
		i += 1


	#calculate the deviation score
	varianceT = variance.transpose()

	deviation = np.zeros((width, width), float)

	deviation = np.dot(varianceT, variance)

	i = 0
	j = 0

	while(i < width):
		j = 0
		while(j < width):
			deviation[j][i] = deviation[j][i]/height
			j += 1
		i += 1

	return deviation, means.transpose()

#Global features classifier

def globalClassifier(globalD):
	
	covMat, mean = getCovarianceandMean(globalD)
	covMatInv = np.linalg.inv(covMat)

	maxThreshold = 0
	for x in globalD:
		threshold = np.dot(np.dot((x - mean), covMatInv),(x - mean).transpose())	
		if threshold > maxThreshold:
			maxThreshold = threshold

		# print np.dot((x - mean), covMatInv)
	print maxThreshold
	return maxThreshold, mean, covMatInv


#Autoencoder
def globalDescriptors(frames, video_name):

	MeanCovarianceFileName = "MeanCovariance/n_MeanCovarianceFile" + video_name + ".npy"
	WeightsFileName = "Weights/n_WeightsFile" + video_name + ".npy"

	thresholdFileName = "Threshold/n_thresholdFile" + video_name 
	thresholdFile = open(thresholdFileName, "wb")

	inputLayerSize  = 500    # side length of sampled image patches
	hiddenLayerSize = 750    # side length of representative image patches
	rho 		= 0.05   # desired average activation of hidden units
	lamda          	= 0.0001 # weight decay parameter
	beta           	= 3      # weight of sparsity penalty term
	max_iterations 	= 400    # number of optimization iterations
	learningRate    = 0.5
		
	# Initialize Neural Network weights randomly

	limit0 = 0
	limit1 = hiddenLayerSize * inputLayerSize
	limit2 = 2 * hiddenLayerSize * inputLayerSize
	limit3 = 2 * hiddenLayerSize * inputLayerSize + hiddenLayerSize
	limit4 = 2 * hiddenLayerSize * inputLayerSize + hiddenLayerSize + inputLayerSize
	
	# W1, W2 values are chosen in the range [-r, r] 

	r = np.sqrt(2.0)/np.sqrt(inputLayerSize + hiddenLayerSize)

	rand = np.random.RandomState(int(time.time()))

	# W1 = np.asarray(rand.uniform(low = 1, high = 1, size = (hiddenLayerSize, inputLayerSize)))
	# W2 = np.asarray(rand.uniform(low = 1, high = 1, size = (inputLayerSize, hiddenLayerSize)))

	# W1 = np.random.randn(inputLayerSize * hiddenLayerSize) * np.sqrt(2.0)/np.sqrt(inputLayerSize + hiddenLayerSize)
	# W2 = np.random.randn(inputLayerSize * hiddenLayerSize) * np.sqrt(2.0)/np.sqrt(inputLayerSize + hiddenLayerSize)
	W1 = np.asarray(rand.uniform(low = -r, high = r, size = (hiddenLayerSize, inputLayerSize)))
	W2 = np.asarray(rand.uniform(low = -r, high = r, size = (inputLayerSize, hiddenLayerSize)))
		
	# Bias values are initialized to zero 
		
	b1 = np.zeros((hiddenLayerSize, 1))
	b2 = np.zeros((inputLayerSize, 1))

	theta = np.concatenate((W1.flatten(), W2.flatten(), b1.flatten(), b2.flatten()))
	
	# inputNodes - Each node is intensity of a patch. (10*10*5 values) 

	numberOfFrames = len(frames)
	m,n = frames[0].shape[:2]
	trainSize = (m/10)*(n/10)*(numberOfFrames/5)  			# number of training examples
	inputNodes = []

	for x in xrange(0, len(frames), 5):
		for i in xrange(0, m, 10):
			for j in xrange(0, n, 10):
				node = []
				for y in xrange(0, 5):
					for k in xrange(0,10):
						for l in xrange(0, 10):
							intensity = frames[x + y][k, l]
							node.append(intensity)
				inputNodes.append(node)

	# print inputNodes
	inputNodes = np.asarray(inputNodes, dtype = np.float32)  #Convert to np array
	inputNodes = np.transpose(inputNodes)
	inputNodes = (inputNodes - inputNodes.mean(axis=0)) / inputNodes.std(axis=0) #Normalization

	numberOfNodes = inputNodes.shape[1]
	oldCost = 0.0
	cost = 0.0

	W1 = theta[limit0 : limit1].reshape(hiddenLayerSize, inputLayerSize)
	W2 = theta[limit1 : limit2].reshape(inputLayerSize, hiddenLayerSize)
	b1 = theta[limit2 : limit3].reshape(hiddenLayerSize, 1)
	b2 = theta[limit3 : limit4].reshape(inputLayerSize, 1)

	hiddenLayer = sigmoid(np.dot(W1, inputNodes) + b1)
	rhoCap = np.sum(hiddenLayer, axis = 1)/numberOfNodes
	tempInputNodes = np.transpose(inputNodes)

	for iter in xrange(max_iterations):
		sumOfSquaresError = 0.0
		weightDecay = 0.0
		sparsityPenalty = 0.0
		for i in xrange(numberOfNodes):

			hiddenLayer = sigmoid(np.dot(W1, np.reshape(tempInputNodes[i], (-1, 1))) + b1)

			outputLayer = sigmoid(np.dot(W2, hiddenLayer) + b2)
			
			diff = outputLayer - np.reshape(tempInputNodes[i], (-1, 1))

			sumOfSquaresError += 0.5 * np.sum(np.multiply(diff, diff)) / tempInputNodes.shape[1]
			weightDecay       += 0.5 * lamda * (np.sum(np.multiply(W1, W1)) + np.sum(np.multiply(W2, W2)))
			sparsityPenalty   += beta * KLDivergence(rho, rhoCap)
			
			KLDivGrad = beta * (-(rho / rhoCap) + ((1 - rho) / (1 - rhoCap)))

			errOut = np.multiply(diff, np.multiply(outputLayer, 1 - outputLayer))
			errHid = np.multiply(np.dot(np.transpose(W2), errOut)  + np.transpose(np.matrix(KLDivGrad)), np.multiply(hiddenLayer, 1 - hiddenLayer))
		
			#Compute the gradient values by averaging partial derivatives
			W2Grad = np.dot(errOut, np.transpose(hiddenLayer))
			W1Grad = np.dot(errHid, np.transpose(np.reshape(tempInputNodes[i], (-1, 1))))
			b1Grad = np.sum(errHid, axis = 1)
			b2Grad = np.sum(errOut, axis = 1)

			#Partial derivatives are averaged over all training examples

			W1Grad = learningRate*(W1Grad / tempInputNodes.shape[1] + lamda * W1)
			W2Grad = learningRate*(W2Grad / tempInputNodes.shape[1] + lamda * W2)
			b1Grad = learningRate*(b1Grad / tempInputNodes.shape[1])
			b2Grad = learningRate*(b2Grad / tempInputNodes.shape[1])	

			W1Grad = np.array(W1Grad)
			W2Grad = np.array(W2Grad)
			b1Grad = np.array(b1Grad)
			b2Grad = np.array(b2Grad)

			# print b2Grad.shape, b2.shape
			W1 = W1 - W1Grad
			W2 = W2 - W2Grad
			b1 = b1 - b1Grad
			b2 = b2 - np.reshape(b2Grad, (-1, 1))

		
		print sumOfSquaresError, weightDecay, sparsityPenalty
		oldCost = cost
		cost = sumOfSquaresError + weightDecay + sparsityPenalty
		if ((cost - oldCost)*(cost - oldCost) < 0.05):
			break	
		print cost

	np.save(WeightsFileName, W1)

	globalD = np.dot(W1, inputNodes).transpose()

	threshold, mean, covMatInv = globalClassifier(globalD)
	
	thresholdFile.write("%f" % threshold)
	np.savez(MeanCovarianceFileName, mean = mean, covMatInv = covMatInv)

	thresholdFile.close()
	return 

def readFrames(directory):
	
	print "Feature extraction Started"
	print "Time : "
	print strftime("%Y-%m-%d %H:%M:%S", gmtime())
	print
	
	videos = []
	
	videos = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory,name)) and '_gt' not in name]
	videos.sort()
	
	for video in videos:
		frames = []
		
		fileNames = [os.path.join(directory,video,name) for name in os.listdir(os.path.join(directory,video)) if os.path.isfile(os.path.join(directory,video,name)) and name.endswith('.tif')]
		fileNames.sort()

		i = 0

		while i < len(fileNames):
			frames.append(cv2.imread(fileNames[i], cv2.IMREAD_GRAYSCALE))
			i += 1
		
		globalDescriptors(frames, video)
	
	print "Feature extraction complete"
	print "Time : "
	print strftime("%Y-%m-%d %H:%M:%S", gmtime())
	return 

if __name__ == "__main__":

	DIR = 'UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train'
	readFrames(DIR)



					



	

