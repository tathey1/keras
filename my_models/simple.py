import keras
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout


def create_model(input_shape,num_classes, args_dict):
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
	except KeyError:
		print('Missing argument[s]')
		print('Arguments must include: dropout_rate, conv_kernels, strides, fc_neurons, and lr')
		return
	if not (len(num_filters) == len(strides) and len(strides) == len(conv_kernels)):
		print('num_filters, strides, and conv_kernels not all the same length')
		return


	
	dropout_rate = args_dict['dropout_rate']
	num_filters = args_dict['num_filters']
	conv_kernels = args_dict['conv_kernels']
	strides = args_dict['strides']
	fc_neurons = args_dict['fc_neurons']
	lr = args_dict['lr']

	model = Sequential()
	model.add(Conv2D(num_filters[0],
                        kernel_size=(conv_kernels[l],conv_kernels[l]),
                        strides=(strides[l],strides[l]),
                        activation='relu', input_shape=input_shape)

	for l in range(1,len(num_filters)):
		model.add(Conv2D(num_filters[l],
			kernel_size=(conv_kernels[l],conv_kernels[l]),
			strides=(strides[l],strides[l]),
			activation='relu'))

	model.add(Flatten())
	model.add(Dense(fc_neurons, activation='relu'))
	model.add(Dropout(dropout_rate))
	model.add(Dense(num_classes,'softmax'))
	optimizer = Adam(lr=lr)
	model.compile(optimizer=optimizer,
		loss=keras.losses.categorical_crossentropy,
		metrics=['accuracy'])
	return model
