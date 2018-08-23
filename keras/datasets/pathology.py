from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..utils.data_utils import get_file
import numpy as np
import os
from PIL import Image
from random import shuffle

def load_data(num_train=360,path='/workspace/data/Part-A_Original/'):
	"""Loads the pathology dataset.
	# Arguments
	path: path to directory that has subdirectories corresponding to each clas
		within these subdirectories are  images
	Returns
	Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`
		x_* is of shape [num_images, num_rows, num_cols,channels]
		y_* is of shape [num_images]
	Dictionary of labels to class names
	"""
	_NUM_TRAIN = num_train
	label_to_class = {}
	subdirs = os.listdir(path)
	label_counter=0
	all_images = []
	all_labels = []
	for subdir in subdirs:
		subdir_name = os.path.join(path,subdir)
    		if os.path.isdir(subdir_name):
			print('Reading images in class: ' + subdir)
			label_to_class[label_counter] = os.path.basename(os.path.normpath(subdir))
			images = os.listdir(subdir_name)
			for image in images:
				image_name = os.path.join(subdir_name,image)
				im = Image.open(image_name)
				imarray = np.array(im)
				imarray = np.expand_dims(imarray,axis=0)
				all_images.append(imarray)
				all_labels.append(np.array([label_counter]))
			label_counter += 1
	index_shuf = range(len(all_images))
	shuffle(index_shuf)
	all_images_shuf = []
	all_labels_shuf = []
	for i in index_shuf:
		all_images_shuf.append(all_images[i])
		all_labels_shuf.append(all_labels[i])
	x_train = np.concatenate(all_images_shuf[:_NUM_TRAIN], axis=0)
	y_train = np.concatenate(all_labels_shuf[:_NUM_TRAIN], axis=0)
	x_test = np.concatenate(all_images_shuf[_NUM_TRAIN:], axis=0)
	y_test = np.concatenate(all_labels_shuf[_NUM_TRAIN:], axis=0)

	return (x_train, y_train), (x_test, y_test), label_to_class

def load_all_data():
	(x_train, y_train), (x_test, y_test), label_to_class = load_data()
	print('Label dictionary:')
	print(label_to_class)
	X = np.append(x_train, x_test,axis=0)
	Y = np.append(y_train, y_test,axis=0)
	return X, Y
