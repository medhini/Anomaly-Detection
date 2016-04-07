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
import pickle

#Autoencoder
def globalDescriptors(frames, video_name):
	
	anomalousFileName = "n_anomalousFile" + video_name
	anomalousFile = open(anomalousFileName, "wb")
	# inputNodes - Each node is intensity of a patch. (10*10*5 values) 

	numberOfFrames = len(frames)
	m,n = frames[0].shape[:2]
	
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
	
	thresholds = []

	"""Read Threshold"""
	directory = 'Threshold'
	thresholdFileNames = [os.path.join(directory,name) for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]
	thresholdFileNames.sort()

	for thresholdFileName in thresholdFileNames:
		thresholdFile = open(thresholdFileName, "rb")
		thresholds.append(float(thresholdFile.read()))

	print thresholds
	"""Read weights"""
	directory = 'Weights'
	weightFileNames = [os.path.join(directory,name) for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]
	weightFileNames.sort()

	"""Read Covariance and Mean"""
	directory = 'MeanCovariance'
	meanCovarianceFileNames = [os.path.join(directory,name) for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]
	meanCovarianceFileNames.sort()

	AnomalousPatches = []

	for i in xrange(len(weightFileNames)):
		
		W1 = np.load(weightFileNames[i])

		# print W1
		data = np.load(meanCovarianceFileNames[i])
		mean = data['mean']
		covMatInv = data['covMatInv']

		# print mean, covMatInv

		globalD = np.dot(W1, inputNodes).transpose()

		"""Classification"""
		for x in xrange(len(globalD)):
			mahalanobisDist = np.dot(np.dot((globalD[x] - mean), covMatInv),(globalD[x] - mean).transpose())
			# print mahalanobisDist
			flag = 0
			for j in xrange(len(thresholds)):
				if mahalanobisDist <= thresholds[j]:
					flag = 1

			if flag == 1:
				continue
			if flag == 0:
				AnomalousPatches.append(x)

	print len(AnomalousPatches)
	pickle.dump(AnomalousPatches, anomalousFile)

	thresholdFile.close()
	# weightFile
	return 

def readFrames(directory):
	
	print "Feature extraction Started"
	print "Time : "
	print strftime("%Y-%m-%d %H:%M:%S", gmtime())
	print
	
	videos = []
	
	videos = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory,name)) and '_gt' not in name]
	videos.sort()
	
	# print videos

	sumOfAllVideos = np.zeros(14)

	globalD = np.zeros((100, 20736))
	
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

	DIR = 'UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test'
	readFrames(DIR)



					



	

