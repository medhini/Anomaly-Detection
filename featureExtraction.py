import cv2
import numpy

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
		for i in xrange(0, m):
			for j in xrange(0,n):

				x = i * 10
				y = j * 8
				
				patch = frame[ i : i + 10, j : j + 8]

				spatialNeighbours = []
				
				#Clock - wise spatialNeighbours starting from left of current patch

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
					descriptors[patchNumber][d] = ssim(neighbour, patch)
					d += 1

				#Temporal Descriptor
				for k in range(frameCount + 1, min(frameCount + 6, len(frames))):
					nextPatch = frames[k][ i : i + 10, j : j + 8]	
					descriptors[patchNumber][d]	= ssim(nextPatch, patch)
					d += 1

				patchNumber += 1

		

def readFrames(fileName):
	
	frames = []
	numberOfFrames = 180
	i = 0

	while i < numberOfFrames:
		frames.append(cv2.imread(fileName))
		i += 1

	localDescriptors(frames)

if __name__ == "__main__":

	x = 'x.jpg'
	readFrames(x)


					



	

