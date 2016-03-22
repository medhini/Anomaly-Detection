import numpy as np
import os

#Inputs: 	fileName->name of the file to read the data from
#			no_of_features->Number of entries in each line (number of dimensions in each sample)

#Returns:	matrix of float type
def getMatrixFromFile(fileName, no_of_features):
	width = no_of_features

	file = open(fileName, "r")
	
	s = []

	for line in file:
		row = line.split(' ')
		row2 = row[0:width]
		s.append(row2)

	s = np.array(s)
	samples = np.zeros(s.shape, float)

	height, width = s.shape

	#Convert the string to float
	i = 0
	j = 0
	while(j < width):
		i = 0
		while(i < height):
			samples[i][j] = float(s[i][j])
			i += 1
		j += 1

	return samples

#Inputs: 	samples->matrix whose cavariance to be found
#			height, width -> dimensions of the matrix

#Returns:	Covariance matrix and the mean vector of the input matrix.
def getCovariance(samples, height, width):

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

descriptorFiles = [f for f in os.listdir("./LocalFeatures/")]

for d in descriptorFiles:
	filePath = os.path.join("./LocalFeatures", d)
	
	s = getMatrixFromFile(filePath, 16)
	h, w = s.shape
	covariance, mean = getCovariance(s, h, w) 
	
	filename_cov = "./Covariances/cov_" + d					#File to store covariance 
	filename_mean = "./Means/mean_" + d						#File to store mean
	
	np.savetxt(filename_cov , covariance, fmt = "%6f")
	np.savetxt(filename_mean , mean, fmt = "%6f")


