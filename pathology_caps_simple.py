'''
Simple CNN model that inspired by capsule net for pathology classification
Iesmantas 2017
'''

from __future__ import print_function
import keras
from keras.datasets import pathology
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
from keras.utils import multi_gpu_model

from keras.preprocessing.image import ImageDataGenerator

import numpy as np

from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adadelta, SGD, Adam


num_classes = 4

# input image dimensions
img_rows, img_cols = 1536, 2048

# the data, split between train and test sets
print('Loading data...')
(x_train, y_train), (x_test, y_test), label_to_class = pathology.load_data()

print('Reformatting data...')
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    im_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    im_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def create_model(input_shape,dropout_rate, fc_neurons):
	model = Sequential()
        model.add(Conv2D(64, kernel_size=(4,4),strides=(2,2),activation='relu',
                        input_shape=input_shape))
	model.add(Conv2D(128, kernel_size=(4,4),strides=(2,2),activation='relu'))
	model.add(Conv2D(256, kernel_size=(6,6),strides=(2,2),activation='relu'))
	model.add(Conv2D(256, kernel_size=(6,6),strides=(2,2),activation='relu'))
	model.add(Conv2D(256, kernel_size=(8,8),strides=(2,2),activation='relu'))


	model.add(Flatten())
        model.add(Dense(fc_neurons, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_classes, activation='softmax'))

        optimizer = Adadelta()

        parallel_model = multi_gpu_model(model,gpus=2)
        parallel_model.compile(optimizer=optimizer,
                               loss=keras.losses.categorical_crossentropy,
                               metrics = ['accuracy'])
        return parallel_model


#new
batch_size = [32]
epochs = [60]
dropout_rate = [0]
fc_neurons = [3872]

datagen = ImageDataGenerator(horizontal_flip=True, validation_split=0.1)

print('Preprocessing images')
training_generator = datagen.flow_from_directory('/workspace/data/Part-A_Original',
	target_size=(512,512),batch_size=36,class_mode='categorical',
	subset='training', interpolation='bilinear')
validation_generator = datagen.flow_from_directory('/workspace/data/Part-A_Original',
        target_size=(512,512),batch_size=40,class_mode='categorical',
        subset='validation', interpolation='bilinear')


print('Creating model')
model = create_model(input_shape=[512,512,3],
	dropout_rate=dropout_rate[0], fc_neurons=fc_neurons[0])

print('Fitting model')
his = model.fit_generator(training_generator,
			epochs=epochs[0], steps_per_epoch=10,
			verbose=1,
			validation_data=validation_generator,
			validation_steps=1)

print('Train accuracy****************************************')
print(his.history['acc'])
print('Validation accuracy************************************')
print(his.history['val_acc'])


total_models = len(batch_size)*len(epochs)*len(dropout_rate)*len(fc_neurons)
counter = 0

