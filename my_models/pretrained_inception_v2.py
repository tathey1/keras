import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import models
from keras import layers
from keras import optimizers

from keras.backend import tf as ktf

class pretrained_inception_v2:
	def __init__(self,args_dict):
		try:
                        if type(args_dict['dropout_rate']) != float:
                                raise NameError('dropout_rate not a float')
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
				'fc_neurons, lr, '+ \
				'input_shape, num_classes, batch_size, '+ \
				'and gpus')
                        return

		self.dropout_rate = args_dict['dropout_rate']
                self.fc_neurons = args_dict['fc_neurons']
                self.lr = args_dict['lr']
                self.input_shape = args_dict['input_shape']
                self.num_classes = args_dict['num_classes']
		self.batch_size = args_dict['batch_size']
		self.gpus = args_dict['gpus']
	
		if self.batch_size % self.gpus != 0:
			print('Batch size cannot be evenly split aross gpus')

	def create_model(self):
		resizer = models.Sequential()
		resizer.add(layers.Lambda(lambda image: ktf.image.resize_images(image,
			(299,299)), input_shape=self.input_shape))


		feature_extractor = InceptionResNetV2(include_top=False,
			input_shape=(299,299,3))
		for layer in feature_extractor.layers:
			layer.trainable = False

		classifier = models.Sequential()
		classifier.add(layers.Flatten())
		classifier.add(layers.Dense(self.fc_neurons,activation='relu'))
		classifier.add(layers.Dropout(self.dropout_rate))
		classifier.add(layers.Dense(self.num_classes,activation='softmax'))

		optimizer = optimizers.Adam(lr=self.lr)

		model = models.Sequential([resizer, feature_extractor,
			classifier])

		if self.gpus > 1:
			parallel_model = multi_gpu_model(model, gpus=self.gpus)
			parallel_model.compile(optimizer=optimizer,
				loss=keras.losses.categorical_crossentropy,
				metrics=['accuracy'])
			model.summary()
			return parallel_model
		else:
			model.compile(optimizer=optimizer,
				loss=keras.losses.categorical_crossentropy,
				metrics=['accuracy'])
			model.summary()
			return model
