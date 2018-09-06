'''
simple model with a procession of convolution layers followed by fully connected layers
'''


import keras
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout, Lambda
from keras.optimizers import Adam
from keras.backend import tf as ktf

class simple:
	def __init__(self,args_dict):
		try:
                        if type(args_dict['dropout_rate']) != float:
                                raise NameError('dropout_rate not a float')
                        if type(args_dict['num_filters']) != list:
                                raise NameError('num_filters not a list')
                        if type(args_dict['conv_kernels']) != list:
                                raise NameError('conv_kernels not a list')
                        if type(args_dict['strides']) != list:
                                raise NameError('strides not a list')
                        if type(args_dict['fc_neurons']) != int:
                                raise NameError('fc_neurons not an int')
                        if type(args_dict['lr']) != float:
                                raise NameError('lr not a float')
			if type(args_dict['input_shape']) != list:
				raise NameError('input_shape is not a list')
			if type(args_dict['num_classes']) != int:
				raise NameError('num_classes is not an int')
                except KeyError:
                        print('Missing argument[s]')
                        print('Arguments must include: dropout_rate, conv_kernels, strides, fc_neurons, lr, input_shape, and num_classes')
                        return

		self.dropout_rate = args_dict['dropout_rate']
                self.num_filters = args_dict['num_filters']
                self.conv_kernels = args_dict['conv_kernels']
                self.strides = args_dict['strides']
                self.fc_neurons = args_dict['fc_neurons']
                self.lr = args_dict['lr']
                self.input_shape = args_dict['input_shape']
                self.num_classes = args_dict['num_classes']
		

                if not (len(self.num_filters) == len(self.strides) and len(self.strides) == len(self.conv_kernels)):
                        print('num_filters, strides, and conv_kernels not all the same length')
                        return
	
	def create_model(self):
		dropout_rate = self.dropout_rate
		num_filters = self.num_filters
		conv_kernels = self.conv_kernels
		strides = self.strides
		fc_neurons = self.fc_neurons
		lr = self.lr
		input_shape = self.input_shape
		num_classes = self.num_classes

		model = Sequential()
		model.add(Lambda(lambda image: ktf.image.resize_images(image,
				(512,512)), input_shape=input_shape))
		model.add(Conv2D(num_filters[0],
                          kernel_size=(conv_kernels[0],conv_kernels[0]),
                          strides=(strides[0],strides[0]),
                          activation='relu'))		
		for l in range(1,len(num_filters)):
			model.add(Conv2D(num_filters[l],
				kernel_size=(conv_kernels[l],conv_kernels[l]),
				strides=(strides[l],strides[l]),
				activation='relu'))

		model.add(Flatten())
		model.add(Dense(fc_neurons, activation='relu'))
		model.add(Dropout(dropout_rate))
		model.add(Dense(num_classes,activation='softmax'))
		optimizer = Adam(lr=lr)
		model.compile(optimizer=optimizer,
			loss=keras.losses.categorical_crossentropy,
			metrics=['accuracy'])
		return model
