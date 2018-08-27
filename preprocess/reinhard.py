'''
author: Thomas Athey
date: 8/27/18

Reinhard preprocessing for keras models

Reference:
Color Transfer between IMages
Erik Reinhard, Michael Ashikhmin, Bruce Gooch, Peter Shirley
Applied Perception Sept/Oct 2001
'''
import math
import numpy as np

def reinhard_normalize(array, stds_target):
	'''
	normalizes the images according to certain lab standard deviations
	Args: array-4d array of shape [num_images, height, width, 3]
		stds-array with 3 stds of lab coordinates
	Returns: array of same shape as array
	'''
	
        rgb = np.moveaxis(array, 3, 0)
        s = rgb.shape
	rgb = np.reshape(rgb, (3,-1))
        


        lms = rgb2lms(rgb)
        del rgb
        c = np.log(10)

        log_lms = np.log(lms)/c
        del lms
        lab = lms2lab(log_lms)
        del log_lms
        
        stds_local = np.std(lab, axis=1)
	stds_local = np.expand_dims(stds_local,axis=1)
	stds_target = np.exapnd_dims(stds_target,axis=1)

	means = np.mean(lab, axis=1)
	means = np.expand_dims(means,axis=1)

	lab = (lab - means)*(stds_target/stds_local) + means
	
	lms = lab2lms(lab)
	del lab
	
	lms = 10**lms
	rgb = lms2rgb(lms)
	del lms

	rgb = np.reshape(rgb, s)
	rgb = np.moveaxis(rgb, 3, 0)
	return rgb


def calculate_stds(array):
	"""
	Calculates the standard deviations
	Args: 4d array of shape [num_images,height, width, 3]
	Returns: array of length 3
	"""

	rgb = np.moveaxis(array, 3, 0)
	rgb = np.reshape(rgb, (3,-1))

	lms = rgb2lms(rgb)
	del rgb
	c = np.log(10)
	log_lms = np.log(lms)/c
	del lms
	lab = lms2lab(log_lms)
	del log_lms

	return np.std(lab, axis=1)

def rgb2lms(rgb):
	"""
	Converts rgb to lms
	Args: 2d array shape [3, num_pixels]
	Returns: 2d array shape [3, num_pixels]
	"""
	print(np.amin(rgb))
	mat = np.array([[0.3811, 0.5783, 0.0402],
		  [0.1967, 0.7244, 0.0782],
		  [0.0241, 0.1288, 0.8444]])
	lms = np.matmul(mat, rgb)

	print(np.amin(lms))
	print(np.argmin(lms))
	return lms

def lms2lab(lms):
	"""
	Converts lms to lab
	Args: 2d array shape [3. num_pixels]
	Returns: 2d array shape [3, num_pixels]
	"""

	mat1 = np.array([[math.sqrt(1./3.),0,0],
		      [0,math.sqrt(1./6.),0],
		      [0,0,math.sqrt(1./2.)]])
	mat2 = np.array([[1.0,1.0,1.0],[1.0,1.0,-2.0],[1.0,-1.0,0]])
	mat = np.matmul(mat1,mat2)

	return np.matmul(mat,lms)	

def lab2lms(lab):
	'''
	Args: 2d array of shape [3, num_pixels]
	Returns: array same shape
	'''
	mat2 = np.array([[math.sqrt(3.)/3.,0,0],
                      [0,math.sqrt(6.)/6.,0],
                      [0,0,math.sqrt(2.)/2.]])
	mat1 = np.array([[1.0,1.0,1.0],[1.0,1.0,-1.0],[1.0,-2.0,0]])
	mat = np.matmul(mat1,mat2)

	return np.matmul(mat,image)

def lms2rgb(lms):
	'''
	Args: 2d array of shape [3, num_pixels]
        Returns: array same shape
        '''

	mat = np.array([[4.4679, -3.5873, 0.1193],
		  [-1.2186, 2.3809, -0.1624],
		  [0.0497, -0.2439, 1.2045]])
	return np.matmul(mat,image)

