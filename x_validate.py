'''
Thomas Athey 9/6/18

performs k fold cross validation. usually called by hyper_param_tune.py

there will be an output directory for each fold, and a results.txt which shows the acc/loss for each fold
each fold directory will include tensorboard events, and a log of acc/loss at each epoch
'''

import keras
from keras.datasets import pathology
from keras import callbacks
import os
import numpy as np
from keras import backend as K
from preprocess import mean_subtract

'''
Args: k - int, number of folds (note that the training/validation partition sizes as determined by k should be divisible by the batch size
the batch size should also be divisible by the number of gpus
out_path-path for the output files
model that implements fucntion create_model which returns a keras model
args_dict - dictionary that specifies batch_size and epochs
'''
def k_fold_xval(k, out_path, model, args_dict):
	try:
		if type(args_dict['batch_size']) != int:
			raise NameError('batch_size not an int')
		if type(args_dict['epochs']) != int:
			raise NameError('epochs not an int')
	except KeyError:
		print('Missing argument[s]')
		print('Arguments must include: batch_size and epochs')
		return
	
	X,Y = pathology.load_all_data()
	Y = keras.utils.to_categorical(Y,np.amax(Y)+1)
	X = X.astype('float32')
	X /= 255

	print('X shape ' + str(X.shape))
	print('Y shape ' + str(Y.shape))

	num_examples = X.shape[0]
	if num_examples % k != 0:
		raise ValueError('Number of folds does not fit into number '+ \
			'of examples: ' +  str(num_examples))
	
	batch_size = args_dict['batch_size']

	num_val = num_examples/k
	num_train = num_examples - num_val
	train_idx = range(num_train)

	results_path = os.path.join(out_path,'results.txt')
	f = open(results_path, 'w')
	f.write('[Loss, accuracy]\n')
	f.close()

	accuracy_cumulative = 0

	for fold in range(k):
		print('Fold #' + str(fold))
		
		out = os.path.join(out_path, str(fold))
		os.makedirs(out)
		print('Results can be found in ' + out)

		x_val = X[num_val*fold:num_val*(fold+1)]
		y_val = Y[num_val*fold:num_val*(fold+1)]
		idxs = [i + num_val*(fold+1) for i in train_idx]
		x_train = X.take(idxs,mode='wrap',axis=0)
		y_train = Y.take(idxs, mode='wrap',axis=0)

		
		print('Preprocessing...')
		(x_train, x_val) = mean_subtract.preprocess(x_train, x_val)
		
		log = callbacks.CSVLogger(out + '/log.csv')
		tb = callbacks.TensorBoard(out + '/tensorboard-logs',
			histogram_freq=1,
			batch_size=batch_size,
			write_graph=False, write_images=False)
		early_stop = callbacks.EarlyStopping(monitor='acc',
			min_delta=0, patience=3, verbose=1)
		print('Creating model...')
		model_instance = model.create_model()

	
		print('Training...')
		model_instance.fit(x_train,y_train,
			batch_size=batch_size,
			epochs=args_dict['epochs'],
			verbose=1,
			validation_data=(x_val, y_val),
			callbacks=[log,early_stop,tb])

		print('Evaluating model...')
		f = open(results_path,'a')
		txt = model_instance.evaluate(x_val, y_val, batch_size)
		print('Loss, accuracy]')
		print(txt)
		f.write(str(txt) + '\n')
		accuracy_cumulative += txt[1]
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

	accuracy_cumulative /= k
	return accuracy_cumulative
