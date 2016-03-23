import cv2
import numpy as np
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import os, os.path
from skimage import color
import csv
from time import gmtime, strftime

def localDescriptors(frames, video_name):

	descriptorFileName = "n_descriptorFile" + video_name
	descriptorFile = open(descriptorFileName, "wb")

	descriptorFile.write("Start time: ")
	descriptorFile.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
	descriptorFile.write("\n")

	MeanCovarianceFile = open("Scene-Wise-MeanCovarianceValues", "wb")
	sumOfAll = np.zeros(14)

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

				for iter in range(8):
					descriptorFile.write("%f " % descriptors[patchNumber][iter])
					
				d = 8

				# Temporal Descriptors

				#Within the patch
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

				#Previous Patch
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

				descriptorFile.write("%f " % descriptors[patchNumber][d])
				d += 1

				#Next Patch
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

				descriptorFile.write("%f " % descriptors[patchNumber][d])


				descriptorFile.write("\n")
				# print descriptors[patchNumber]
				# print len(descriptors[patchNumber]), len(sumOfAll)
				sumOfAll = np.sum((descriptors[patchNumber], sumOfAll), axis = 0)
				patchNumber += 1
	
	sumOfAll = np.divide(sumOfAll, (len(frames)/5) * 36 * 24)
	return  sumOfAll

def readFrames(directory):
	
	print "Feature extraction Started"
	print "Time : "
	print strftime("%Y-%m-%d %H:%M:%S", gmtime())
	print
	
	videos = []
	
	videos = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory,name)) and '_gt' not in name]
	videos.sort()
	
	print videos

	sumOfAllVideos = np.zeros(14)
	
	for video in videos:
		frames = []
		#fileNames = [os.path.join(directory,name) for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]
		fileNames = [os.path.join(directory,video,name) for name in os.listdir(os.path.join(directory,video)) if os.path.isfile(os.path.join(directory,video,name)) and name.endswith('.tif')]
		fileNames.sort()

		i = 0

		while i < len(fileNames):
			frames.append(cv2.imread(fileNames[i]))
			i += 1
		
		sumOfAllVideos = np.sum((localDescriptors(frames, video), sumOfAllVideos), axis=0)
	
	print np.divide(sumOfAllVideos, 16)

	# Mean - [ 3.56797736  3.33903645  3.86414657  3.30907188  3.56797736  3.33903645 3.86414657  3.30907188  0.98170552  0.98172827  0.98177407  0.98182942 4.52607493  4.37725463]
	
	print "Feature extraction complete"
	print "Time : "
	print strftime("%Y-%m-%d %H:%M:%S", gmtime())
	
	return 

if __name__ == "__main__":

	DIR = 'UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train'
	readFrames(DIR)
