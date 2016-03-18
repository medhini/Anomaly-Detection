import cv2
import numpy as np
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import os, os.path
from skimage import color
import csv
from time import gmtime, strftime
import json

def localize(anomalousFrame):
	return

def getFrameClass(frames, video_name):

	classLabelFileName = "n_classLabelFile" + video_name
	classLabelFileName = open(classLabelFileName, "wb")
		
	normal = []
	anomalous = {}

	iter = 1
	for frame in frames:
		flag = 0
		for x in xrange(frame.shape[0]):
			for y in xrange(frame.shape[1]):
				pixel = frame[x][y]
				if pixel[0] > 0 or pixel[1] > 0 or pixel[2] > 0:
					flag = 1
					anomalous[iter] = pixel
		
		if flag == 0:
			normal.append(iter)
		
		iter += 1
	
	for item in normal:
		classLabelFileName.write("%s\n" % item)
	
	classLabelFileName.write(str(anomalous))
	classLabelFileName.close()
	
	return

def readGTFrames(directory):
	
	print "Formulating Ground truth Started"
	print "Time : "
	print strftime("%Y-%m-%d %H:%M:%S", gmtime())
	print
	
	videos = []
	
	videos = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory,name)) and '_gt' in name]
	videos.sort()
	
	print videos
	
	for video in videos:
		frames = []
		#fileNames = [os.path.join(directory,name) for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]
		fileNames = [os.path.join(directory,video,name) for name in os.listdir(os.path.join(directory,video)) if os.path.isfile(os.path.join(directory,video,name)) and name.endswith('.bmp')]
		fileNames.sort()

		i = 0

		while i < len(fileNames):
			frames.append(cv2.imread(fileNames[i]))
			i += 1
		
		getFrameClass(frames, video)
	
	print "Feature extraction complete"
	print "Time : "
	print strftime("%Y-%m-%d %H:%M:%S", gmtime())
	print 

	return 

if __name__ == "__main__":

	DIR = 'UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test'
	readGTFrames(DIR)