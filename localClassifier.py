import cv2
import os, os.path

def computeMeanCovariance(descriptorFileName):

	descriptorFile = open(descriptorFileName, "r")

	for line in descriptorFile.readline():
		
	
def readDatapoints(directory):
	
	fileNames = [os.path.join(directory,name) for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]
	# fileNames = [os.path.join(directory,fileName) for name in os.listdir(os.path.join(directory,video)) if os.path.isfile(os.path.join(directory,video,name)) and name.endswith('.tif')]
	fileNames.sort()

	for name in fileNames:
		computeMeanCovariance(name)
	return 

if __name__ == "__main__":
	DIR = 'LocalFeatures'
	readDatapoints(DIR)