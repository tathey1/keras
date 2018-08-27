import keras
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout, Lambda, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.backend import tf as ktf
from keras.utils import multi_gpu_model

class max_pool:
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
			if type(args_dict['batch_size']) != int:
				raise NameError('batch_size is not an int')
			if type(args_dict['gpus']) != int:
				raise NameError('gpus is not and int')
                except KeyError:
                        print('Missing argument[s]')
                        print('Arguments must include: dropout_rate, '+ \
				'conv_kernels, strides, fc_neurons, lr, '+ \
				'input_shape, num_classes, batch_size, '+ \
				'and gpus')
                        return

		self.dropout_rate = args_dict['dropout_rate']
                self.num_filters = args_dict['num_filters']
                self.conv_kernels = args_dict['conv_kernels']
                self.strides = args_dict['strides']
                self.fc_neurons = args_dict['fc_neurons']
                self.lr = args_dict['lr']
                self.input_shape = args_dict['input_shape']
                self.num_classes = args_dict['num_classes']
		self.batch_size = args_dict['batch_size']
		self.gpus = args_dict['gpus']

                if not (len(self.num_filters) == len(self.strides) and len(self.strides) == len(self.conv_kernels)):
                        print('num_filters, strides, and conv_kernels not '+ \
				'all the same length')
                        return
		if self.batch_size % self.gpus != 0:
			print('Batch size cannot be evenly split aross gpus')
	

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
		#model.add(Lambda(lambda image: ktf.image.resize_images(image,
		#		(512,512)), input_shape=input_shape))
		model.add(Lambda(self._tile_images, input_shape=input_shape))
		for l in range(len(num_filters)):
			model.add(Conv2D(num_filters[l],
				kernel_size=(conv_kernels[l],conv_kernels[l]),
				strides=(strides[l],strides[l])))
			model.add(BatchNormalization())
			model.add(Activation('relu'))
		model.add(Lambda(self._max_tile))
		model.add(Flatten())
		model.add(Dense(fc_neurons, activation='relu'))
		model.add(Dropout(dropout_rate))
		model.add(Dense(num_classes,activation='softmax'))
		optimizer = Adam(lr=lr)


		if self.gpus > 1:
			parallel_model = multi_gpu_model(model, gpus=self.gpus)
			parallel_model.compile(optimizer=optimizer,
				loss=keras.losses.categorical_crossentropy,
				metrics=['accuracy'])
			return parallel_model
		else:
			model.compile(optimizer=optimizer,
				loss=keras.losses.categorical_crossentropy,
				metrics=['accuracy'])
			return model
	


	def _tile_images(self, images):
		#num_images = args['num_images']
		num_images = self.batch_size/self.gpus
		im_list = ktf.split(images,num_images,0)
		tile_1 = im_list
		counter=0
		for im in im_list:
			temp = ktf.split(im,3,1)
			tile_1[counter]=ktf.concat(temp,0)
			counter+=1

		tile_2=im_list
		counter=0
		for im in tile_1:
			temp = ktf.split(im,4,2)
			tile_2[counter] = ktf.concat(temp,0)
			counter+=1
		return ktf.concat(tile_2,0)

	def _max_tile(self, images):
		#num_images = args['num_images']
		num_images = self.batch_size/self.gpus
		im_list = ktf.split(images, num_images,0)
		maxed = im_list
		counter=0
		for im in im_list:
			maxed[counter] = ktf.reduce_max(im,axis=0,keepdims=True)
			counter+=1
		return ktf.concat(maxed,0)
