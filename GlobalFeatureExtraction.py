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
# from covariance.py import * 

def sigmoid(x):
	return 1/(1 + np.exp(-x))

def KLDivergence(rho, rhoCap):
	return np.sum(rho*np.log(rho/rhoCap) + (1 - rho)/np.log((1 - rho)/ (1 - rhoCap)))

#Covariance
def getCovarianceandMean(samples):

	height, width = samples.shape

	print height
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

	print width
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

	return deviation, means

#Global features classifier

def globalClassifier(globalD):
	
	covMat, mean = getCovarianceandMean(globalD)
	covMatInv = np.linalg.inv(covMat)

	
	threshold = np.amax(np.multiply(np.multiply((globalD - mean).transpose(), covMatInv),(globalD - mean)))
	
	print threshold
	return


#Autoencoder
def globalDescriptors(frames):
	
	inputLayerSize  = 500    # side length of sampled image patches
	hiddenLayerSize = 100    # side length of representative image patches
	rho 						= 0.05   # desired average activation of hidden units
	lamda          	= 0.0001 # weight decay parameter
	beta           	= 3      # weight of sparsity penalty term
	max_iterations 	= 400    # number of optimization iterations
		
	# Initialize Neural Network weights randomly

	limit0 = 0
	limit1 = hiddenLayerSize * inputLayerSize
	limit2 = 2 * hiddenLayerSize * inputLayerSize
	limit3 = 2 * hiddenLayerSize * inputLayerSize + hiddenLayerSize
	limit4 = 2 * hiddenLayerSize * inputLayerSize + hiddenLayerSize + inputLayerSize
	
	# W1, W2 values are chosen in the range [-r, r] 

	# r = np.sqrt(1) / np.sqrt(inputLayerSize + hiddenLayerSize + 1)

	r = 90
	rand = np.random.RandomState(int(time.time()))

	# W1 = np.asarray(rand.uniform(low = -1/r, high = r, size = (hiddenLayerSize, inputLayerSize)))
	# W2 = np.asarray(rand.uniform(low = -1/r, high = r, size = (inputLayerSize, hiddenLayerSize)))

	W1 = np.asarray(1, size = (hiddenLayerSize, inputLayerSize)))
	W2 = np.asarray(1, size = (inputLayerSize, hiddenLayerSize)))
		
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

	inputNodes = np.asarray(inputNodes, dtype = np.float32)  #Convert to np array
	inputNodes = np.transpose(inputNodes)
	inputNodes = (inputNodes - inputNodes.mean(axis=0)) / inputNodes.std(axis=0) #Normalization

	for iter in xrange(max_iterations):

		W1 = theta[limit0 : limit1].reshape(hiddenLayerSize, inputLayerSize)
		W2 = theta[limit1 : limit2].reshape(inputLayerSize, hiddenLayerSize)
		b1 = theta[limit2 : limit3].reshape(hiddenLayerSize, 1)
		b2 = theta[limit3 : limit4].reshape(inputLayerSize, 1)

		hl = np.dot(W1, inputNodes) + b1
		
		hiddenLayer = sigmoid(np.dot(W1, inputNodes) + b1)
		outputLayer = sigmoid(np.dot(W2, hiddenLayer) + b2)

		# for i in range(len(hiddenLayer)):
		# 	if np.all(hiddenLayer[i] == 0):
		# 		hiddenLayer[i] += 0.0001

		rhoCap = np.sum(hiddenLayer, axis = 1)/numberOfFrames

		# print len(outputLayer), len(inputNodes)
		diff = outputLayer - inputNodes

		# print np.dot(W2, hiddenLayer) + b2

		sumOfSquaresError = 0.5 * np.sum(np.multiply(diff, diff)) / numberOfFrames
		weightDecay       = 0.5 * lamda * (np.sum(np.multiply(W1, W1)) + np.sum(np.multiply(W2, W2)))
		sparsityPenalty   = beta * KLDivergence(rho, rhoCap)
		cost              = sumOfSquaresError + weightDecay + sparsityPenalty
		
		KLDivGrad = beta * (-(rho / rhoCap) + ((1 - rho) / (1 - rhoCap)))

		delOut = np.multiply(diff, np.multiply(outputLayer, 1 - outputLayer))
		delHid = np.multiply(np.dot(np.transpose(W2), delOut) + np.transpose(np.matrix(KLDivGrad)), np.multiply(hiddenLayer, 1 - hiddenLayer)) 
		
		# print outputLayer
		 # + np.transpose(np.matrix(KLDivGrad))
		#Compute the gradient values by averaging partial derivatives
			
		W1Grad = np.dot(delHid, np.transpose(inputNodes))
		W2Grad = np.dot(delOut, np.transpose(hiddenLayer))
		b1Grad = np.sum(delHid, axis = 1)
		b2Grad = np.sum(delOut, axis = 1)
		
		#Partial derivatives are averaged over all training examples

		W1Grad = W1Grad / numberOfFrames + lamda * W1
		W2Grad = W2Grad / numberOfFrames + lamda * W2
		b1Grad = b1Grad / numberOfFrames
		b2Grad = b2Grad / numberOfFrames	

		W1 = np.array(W1Grad)
		W2 = np.array(W2Grad)
		b1 = np.array(b1Grad)
		b2 = np.array(b2Grad)

		theta = np.concatenate((W1.flatten(), W2.flatten(), b1.flatten(), b2.flatten()))
				
		print sumOfSquaresError	

	# W1 = theta[limit0 : limit1].reshape(hiddenLayerSize, inputLayerSize)
	# globalD = np.dot(W1, inputNodes).transpose()

	# globalClassifier(globalD)
	
	# np.save('finalWeights', W1)
	# np.savetxt('globalDescriptors.txt', globalD)

	return 

def readFrames(directory):
	
	frames = []
	fileNames = [os.path.join(directory,name) for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]
	fileNames.sort()

	i = 0

	while i < len(fileNames):
		frames.append(cv2.imread(fileNames[i], cv2.IMREAD_GRAYSCALE))
		i += 1

	globalDescriptors(frames)
	return 

if __name__ == "__main__":

	DIR = 'UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/Train001'
	readFrames(DIR)



					



	

