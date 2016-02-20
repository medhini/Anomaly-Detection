import cv2
import numpy 
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import os, os.path
from skimage import color
import csv

def localDescriptors(frames):

	frameCount = 0
	descriptorFile = open("descriptorFile", "wb")
	for frame in frames:
		frameCount += 1
		m,n = frame.shape[:2]

		m = m / 10
		n = n / 8

		descriptors = numpy.zeros((m * n, 13)) 
		patchNumber = 0
	
		descriptorFile.write("%d\n" % frameCount)

		for x in xrange(0, m):
			for y in xrange(0,n):

				descriptorFile.write("%d : " % patchNumber)
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

				# Spatial Descriptor
				for neighbour in spatialNeighbours:
					patch = color.rgb2gray(patch)
					neighbour = color.rgb2gray(neighbour)
					if(patch.shape == neighbour.shape):
						# print patch.shape, neighbour.shape
						descriptors[patchNumber][d] = ssim(neighbour, patch)
						
					else :
						descriptors[patchNumber][d] = 0	

					descriptorFile.write("%f " % descriptors[patchNumber][d])
					d += 1

				# Temporal Descriptor
				for k in range(frameCount + 1, frameCount + 6):
					
					if k < len(frames):
						nextPatch = frames[k][ i : i + 10, j : j + 8]	
						if(nextPatch.shape == patch.shape):
							patch = color.rgb2gray(patch)
							nextPatch = color.rgb2gray(nextPatch)
							descriptors[patchNumber][d]	= ssim(nextPatch, patch)
						else :
							descriptors[patchNumber][d] = 0
					else :
						descriptors[patchNumber][d] = 0
					
					descriptorFile.write("%f " % descriptors[patchNumber][d])
					d += 1

				descriptorFile.write("\n")
				patchNumber += 1
	return 

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

	localDescriptors(frames)
	return 

if __name__ == "__main__":

	DIR = 'UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/Train001'
	readFrames(DIR)

	# for x in xrange(len(d)):
	# 	for y in xrange(len(d[0])):
	# 		print d[x][y],



					



	

