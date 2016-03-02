import cv2
import numpy 
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import os, os.path
from skimage import color
import csv

def localDescriptors(frames):

	descriptorFile = open("descriptorFile", "wb")
	for frameNumber in range(0, len(frames), 5):
		
		m,n = frames[frameNumber].shape[:2]

		m = m / 10
		n = n / 8

		descriptors = numpy.zeros((m * n, 14)) 
		patchNumber = 0
	
		descriptorFile.write("%d\n" % frameNumber)
			
		for x in xrange(0, m):
			for y in xrange(0,n):
				descriptorFile.write("%d : " % patchNumber)
				i = x * 10
				j = y * 8

				# Spatial Descriptor
				for frame in range(frameNumber, frameNumber + 5, 1):
					
					d = 0 #Descriptor number
					patch = frames[frame][ i : i + 10, j : j + 8]
					spatialNeighbours = []
					
					#AntiClock - wise spatialNeighbours starting from left of current patch

					spatialNeighbours.append(frames[frame][ i - 10 : i, j : j + 8])
					spatialNeighbours.append(frames[frame][ i - 10 : i, j + 8 : j + 16])
					spatialNeighbours.append(frames[frame][ i : i + 10, j + 8 : j + 16])
					spatialNeighbours.append(frames[frame][ i + 10 : i + 20, j + 8 : j + 16])
					spatialNeighbours.append(frames[frame][ i + 10 : i + 20, j : j + 8])
					spatialNeighbours.append(frames[frame][ i + 10 : i + 20, j - 8 : j])
					spatialNeighbours.append(frames[frame][ i : i + 10, j - 8 : j])
					spatialNeighbours.append(frames[frame][ i - 10 : i, j - 8 : j])

					for neighbour in spatialNeighbours:
						patch = color.rgb2gray(patch)
						neighbour = color.rgb2gray(neighbour)
						if(patch.shape == neighbour.shape):
							# print patch.shape, neighbour.shape
							descriptors[patchNumber][d] += ssim(neighbour, patch)
						d += 1

				for iter in range(8):
					descriptors[patchNumber][iter] /= 5
					descriptorFile.write("%f " % descriptors[patchNumber][iter])
					
				d = 8
				# Temporal Descriptors
				for frame in range(frameNumber, frameNumber + 4, 1):
					if frame + 1 < len(frames):
						patch = frames[frame][i : i + 10, j : j + 8]
						nextPatch = frames[frame + 1][i : i + 10, j : j + 8]
						patch = color.rgb2gray(patch)
						nextPatch = color.rgb2gray(nextPatch)	
						if(nextPatch.shape == patch.shape):
							descriptors[patchNumber][d]	= ssim(nextPatch, patch)
					
					descriptorFile.write("%f " % descriptors[patchNumber][d])
					d += 1

				if frameNumber - 5 >= 0:
					for frame in range(frameNumber - 5, frameNumber):
						patch1 = frames[frame][i : i + 10, j : j + 8]
						patch2 = frames[frame + 5][i : i + 10, j : j + 8]
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
						patch1 = frames[frame][i : i + 10, j : j + 8]
						patch2 = frames[frame + 5][i : i + 10, j : j + 8]
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

def globalDescriptors(frames):

	# cubes = []
	# for i in range(0, len(frames), 5):

	# 	for j in range(i, i + 5, 1):
						


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



					



	

