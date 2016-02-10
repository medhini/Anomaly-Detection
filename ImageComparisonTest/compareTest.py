import cv2
import numpy as np
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt

def mean_squared_error(imageA, imageB):
	error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	error /= float(imageA.shape[0] * imageA.shape[1])
	
	return error
	
def compare(imageA, imageB):
	m = mean_squared_error(imageA, imageB)
	s = ssim(imageA, imageB)
	
	print m
	print s

#Testing part hereafter. Not Important

#Read and resize image1	
image1 = cv2.imread('image1.jpg')
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image1 = cv2.resize(image1, (100, 100), interpolation = cv2.INTER_CUBIC)

#Read and resize image2
image2 = cv2.imread('image2.jpg')
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
image2 = cv2.resize(image2, (100, 100), interpolation = cv2.INTER_CUBIC)

#Compare the two images. Performing both Mean Squared error and Structural Similarity Index
compare(image1, image2)

#Show the two images just for visual comparison
cv2.imshow('image1', image1)
cv2.imshow('image2', image2)

cv2.waitKey(0)	
cv2.destroyAllWindows()
