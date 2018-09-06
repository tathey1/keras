'''
Thomas Athey 9/7/18

The function eval defined here simply evaluates a model against a single fold of the data
This is done to leverage the data augmentation options in ImageDataGenerator

It would be nice to be able to use ImageDataGenerator with multiple folds

eval creates a tensorboard event file, a log.txt which has the losses/accuracies at each epoch
and results.txt which has the final loss/accuracy of the training
'''

import keras
from keras.datasets import pathology
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
import os
import numpy as np
from keras import backend as K
from preprocess import mean_subtract

'''
Args: out_path - path for where the output files should go
model - object that has create model function that returns a keras model
args_dict - arguments to specify batch size and epochs for training
'''
def eval(out_path, model, args_dict):

	try:
		if type(args_dict['batch_size']) != int:
			raise NameError('batch_size not an int')
		if type(args_dict['epochs']) != int:
			raise NameError('epochs not an int')
	except KeyError:
		print('Missing argument[s]')
		print('Arguments must include: batch_size and epochs')
		return
	
	(x_train, y_train), (x_val, y_val), label_dict = pathology.load_data()
	print(label_dict)
	num_classes = np.amax(y_train)+1 #assumption
	y_train = keras.utils.to_categorical(y_train,num_classes)
	y_val = keras.utils.to_categorical(y_val,num_classes)
	x_train = x_train.astype('float32')
	x_val = x_val.astype('float32')
	x_train /= 255
	x_val /= 255

	print('X train shape ' + str(x_train.shape))
	print('Y train shape ' + str(y_train.shape))

	batch_size = args_dict['batch_size']
	epochs = args_dict['epochs']
	
	results_path = os.path.join(out_path,'results.txt')
	f = open(results_path, 'w')
	f.write('[Loss, accuracy]\n')
	f.close()
	
	datagen = ImageDataGenerator(
		featurewise_center=False,
		featurewise_std_normalization=False,
		rotation_range=45,
		width_shift_range=0.1,
		height_shift_range=0.1,
		horizontal_flip=True)
	print('Preprocessing...')
	datagen.fit(x_train)
	
	

	log = callbacks.CSVLogger(out_path + '/log.csv')
	tb = callbacks.TensorBoard(out_path + '/tensorboard-logs',
		histogram_freq=1,
		batch_size=batch_size,
		write_graph=False, write_images=False)
	early_stop = callbacks.EarlyStopping(monitor='loss',
		min_delta=0, patience=5, verbose=1)
	print('Creating model...')
	model_instance = model.create_model()
	#model_instance.summary()
	

	print('Training...')
	model_instance.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(x_train)/batch_size, epochs=epochs,
		validation_data=(x_val,y_val), callbacks=[log,tb]) #no early stop

	print('Evaluating model...')
	f = open(results_path,'a')
	txt = model_instance.evaluate(x_val, y_val, batch_size)
	print('Loss, accuracy]')
	print(txt)
	f.write(str(txt) + '\n')
	accuracy =  txt[1]
	f.close()

	print('Using model for prediction...')
	f = open(os.path.join(out_path,'predict.txt'),'a')
	predictions = str(model_instance.predict(x_val, args_dict['batch_size']))
	predictions = predictions.replace('[', '')
	predictions = predictions.replace(']', '')
	f.write(predictions+'\n')
	f.close()

	f = open(os.path.join(out_path,'truth.txt'),'a')
	truth = str(y_val)
	truth = truth.replace('[','')
	truth = truth.replace(']','')
	f.write(truth+'\n')
	f.close()
	K.clear_session()

	return accuracy
