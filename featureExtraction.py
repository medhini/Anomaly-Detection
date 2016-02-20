import cv2
import numpy 
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import os, os.path
from skimage import color

def localDescriptors(frames):

	frameCount = 0

	for frame in frames:
		frameCount += 1
		m,n = frame.shape[:2]

		m = m / 10
		n = n / 8

		descriptors = numpy.zeros((m * n, 13)) 
		patchNumber = 0

		#Spatial Descriptor
		for x in xrange(0, m):
			for y in xrange(0,n):

				i = x * 10
				j = y * 8
				
				patch = frame[ i : i + 10, j : j + 8]

				spatialNeighbours = []
				
				#AntiClock - wise spatialNeighbours starting from left of current patch

				spatialNeighbours.append(frame[ i - 10 : i, j : j + 8])
				spatialNeighbours.append(frame[ i - 10 : i, j + 8 : j + 16])
				spatialNeighbours.append(frame[ i : i + 10, j + 8 : j + 16])
				spatialNeighbours.append(frame[ i + 10 : i + 20, j + 8 : j + 16])
				spatialNeighbours.append(frame[ i + 10 : i + 20, j : j + 8])
				spatialNeighbours.append(frame[ i + 10 : i + 20, j - 8 : j])
				spatialNeighbours.append(frame[ i : i + 10, j - 8 : j])
				spatialNeighbours.append(frame[ i - 10 : i, j - 8 : j])

				d = 0 #Descriptor number

				for neighbour in spatialNeighbours:
					if(patch.shape == neighbour.shape):
						patch = color.rgb2gray(patch)
						neighbour = color.rgb2gray(neighbour)
						# print patch.shape, neighbour.shape
						descriptors[patchNumber][d] = ssim(neighbour, patch)
					else :
						descriptors[patchNumber][d] = 0	
					d += 1

				# Temporal Descriptor
				for k in range(frameCount + 1, min(frameCount + 6, len(frames))):
					nextPatch = frames[k][ i : i + 10, j : j + 8]	
					if(nextPatch.shape == patch.shape):
						patch = color.rgb2gray(patch)
						nextPatch = color.rgb2gray(nextPatch)
						descriptors[patchNumber][d]	= ssim(nextPatch, patch)
					else :
						descriptors[patchNumber][d] = 0
					d += 1

				patchNumber += 1
				# print patchNumber
		print frameCount
	return descriptors

def globalDescriptors(frames):
	return 

def readFrames(directory):
	
	frames = []
	fileNames = [os.path.join(directory,name) for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]
	fileNames.sort()

	i = 0

	while i < len(fileNames):
		frames.append(cv2.imread(fileNames[i]))
		i += 1

	print frames
	localD = localDescriptors(frames)

	return localD

if __name__ == "__main__":

	DIR = 'UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/Train001'
	d = readFrames(DIR)

	# for x in xrange(len(d)):
	# 	for y in xrange(len(d[0])):
	# 		print d[x][y],



					



	

