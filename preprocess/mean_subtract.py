'''
author: Thomas Athey
date: 8/27/18

preprocess by simply subtracting mean of rgb channels
'''
import numpy as np

def preprocess(train, val):
	'''
	train and val both 4d arrays of shape [num_images, height, width, 3]
	'''
	train = np.swapaxes(train,3,0)
	shape_train = train.shape
		
	val = np.swapaxes(val, 3, 0)
	shape_val = val.shape

	train = np.reshape(train,[3,-1])
	val = np.reshape(val, [3,-1])

	means = np.mean(train, axis=1)
	means = np.expand_dims(means,axis=1)
	stds = np.std(train, axis=1)
	stds = np.expand_dims(stds, axis=1)
	
	train = (train - means)/stds
	val = (val - means)/stds

	train = np.reshape(train, shape_train)
	val = np.reshape(val, shape_val)

	train = np.swapaxes(train, 3, 0)
	val = np.swapaxes(val, 3, 0)

	return train, val
