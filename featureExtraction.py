import cv2
import numpy as np 
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import os, os.path
from skimage import color
import csv

def localDescriptors(frames):

	descriptorFile = open("descriptorFile-test", "wb")
	for frameNumber in range(0, len(frames), 5):
		
		m,n = frames[frameNumber].shape[:2]

		m = m / 10
		n = n / 10

		descriptors = np.zeros((m * n, 14)) 
		patchNumber = 0
	
		descriptorFile.write("%d\n" % frameNumber)
			
		for x in xrange(0, m):
			for y in xrange(0,n):
				descriptorFile.write("%d : " % patchNumber)
				i = x * 10
				j = y * 10

				# Spatial Descriptor
				for frame in range(frameNumber, frameNumber + 5, 1):
					
					d = 0 #Descriptor number
					patch = frames[frame][ i : i + 10, j : j + 10]
					spatialNeighbours = []
					
					#AntiClock - wise spatialNeighbours starting from left of current patch

					spatialNeighbours.append(frames[frame][ i - 10 : i, j : j + 10])
					spatialNeighbours.append(frames[frame][ i - 10 : i, j + 10 : j + 20])
					spatialNeighbours.append(frames[frame][ i : i + 10, j + 10 : j + 20])
					spatialNeighbours.append(frames[frame][ i + 10 : i + 20, j + 10 : j + 20])
					spatialNeighbours.append(frames[frame][ i + 10 : i + 20, j : j + 10])
					spatialNeighbours.append(frames[frame][ i + 10 : i + 20, j - 10 : j])
					spatialNeighbours.append(frames[frame][ i : i + 10, j - 10 : j])
					spatialNeighbours.append(frames[frame][ i - 10 : i, j - 10 : j])

					for neighbour in spatialNeighbours:
						patch = color.rgb2gray(patch)
						neighbour = color.rgb2gray(neighbour)
						if(patch.shape == neighbour.shape):
							# print patch.shape, neighbour.shape
							descriptors[patchNumber][d] += ssim(neighbour, patch)
						d += 1

				for iter in range(10):
					descriptors[patchNumber][iter] /= 5
					descriptorFile.write("%f " % descriptors[patchNumber][iter])
					
				d = 8
				# Temporal Descriptors
				for frame in range(frameNumber, frameNumber + 4, 1):
					if frame + 1 < len(frames):
						patch = frames[frame][i : i + 10, j : j + 10]
						nextPatch = frames[frame + 1][i : i + 10, j : j + 10]
						patch = color.rgb2gray(patch)
						nextPatch = color.rgb2gray(nextPatch)	
						if(nextPatch.shape == patch.shape):
							descriptors[patchNumber][d]	= ssim(nextPatch, patch)
					
					descriptorFile.write("%f " % descriptors[patchNumber][d])
					d += 1

				if frameNumber - 5 >= 0:
					for frame in range(frameNumber - 5, frameNumber):
						patch1 = frames[frame][i : i + 10, j : j + 10]
						patch2 = frames[frame + 5][i : i + 10, j : j + 10]
						patch1 = color.rgb2gray(patch1)
						patch2 = color.rgb2gray(patch2)
						if (patch1.shape == patch2.shape):
							descriptors[patchNumber][d]	+= ssim(patch1, patch2) 
						else :
							descriptors[patchNumber][d]	+= 0

				descriptors[patchNumber][d] /= 5
				descriptorFile.write("%f " % descriptors[patchNumber][d])
				d += 1

				if frameNumber + 10 < len(frames):
					for frame in range(frameNumber, frameNumber + 5):
						patch1 = frames[frame][i : i + 10, j : j + 10]
						patch2 = frames[frame + 5][i : i + 10, j : j + 10]
						patch1 = color.rgb2gray(patch1)
						patch2 = color.rgb2gray(patch2)
						if (patch1.shape == patch2.shape):
							descriptors[patchNumber][d]	+= ssim(patch1, patch2)
						else :
							descriptors[patchNumber][d]	+= 0

				descriptors[patchNumber][d] /= 5
				descriptorFile.write("%f " % descriptors[patchNumber][d])
				d += 1

				descriptorFile.write("\n")
				patchNumber += 1
	return 

def sigmoid(x):
	return 1/(1+np.exp(-x))

def KL_Divergence(rho, rho1):
	return rho/np.log(rho/rho1) + (1 - rho)/np.log((1 - rho)/ (1 - rho1))

def globalDescriptors(frames):
	
	inputNodesSize  = 3      # side length of sampled image patches
	hiddenNodesSize = 5      # side length of representative image patches
	rho            	= 0.01   # desired average activation of hidden units
	lamda          	= 0.0001 # weight decay parameter
	beta           	= 3      # weight of sparsity penalty term
	num_patches    	= 10000  # number of training examples
	max_iterations 	= 400    # number of optimization iterations	
    
    # Initialize Neural Network weights randomly
    # W1, W2 values are chosen in the range [-r, r] 

	r = np.sqrt(6) / np.sqrt(inputNodesSize + hiddenNodesSize + 1)

	rand = numpy.random.RandomState(int(time.time()))
    
	W1 = numpy.asarray(rand.uniform(low = -r, high = r, size = (hiddenNodesSize, inputNodesSize)))
	W2 = numpy.asarray(rand.uniform(low = -r, high = r, size = (inputNodesSize, hiddenNodesSize)))
    
    # Bias values are initialized to zero 
    
	b1 = numpy.zeros((hiddenNodesSize, 1))
	b2 = numpy.zeros((inputNodesSize, 1))

    # Create 'theta' by unrolling W1, W2, b1, b2 

	theta = numpy.concatenate((W1.flatten(), W2.flatten(), b1.flatten(), b2.flatten()))
    
    #inputNodes - Each node is intensity of a patch. (10*10*5 values) 

	inputNodes = []

	m,n = frames[0].shape[:2]

	for x in xrange(0, len(frames), 5):
		for i in xrange(0, m, 10):
			for j in xrange(0, n, 10):
				node = []
				for y in xrange(1, 5):
					for l in xrange(0,10):
						for m in xrange(0, 10):
							intensity = frames[y][l, m]
							node.extend(intensity)
				inputNodes.append(node)

	print inputNodes

    # #Feedforward
    # z1 = np.dot(W1, inputNodes) + b1
    # o1 = sigmoid(z1)
    # z2 = np.dot(o1, W2)


	return 

def readFrames(directory):
	
	frames = []
	fileNames = [os.path.join(directory,name) for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]
	fileNames.sort()

	i = 0

	while i < len(fileNames):
		frames.append(cv2.imread(fileNames[i]))
		i += 1

	localDescriptors(frames)
	globalDescriptors(frames)
	return 

if __name__ == "__main__":

	DIR = 'UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/Train001'
	readFrames(DIR)

	# for x in xrange(len(d)):
	# 	for y in xrange(len(d[0])):
	# 		print d[x][y],



					



	

