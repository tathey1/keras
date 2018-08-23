'''
Simple CNN model that inspired by capsule net for pathology classification
Iesmantas 2017
'''

from __future__ import print_function
import keras
from keras.datasets import pathology
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Lambda
from keras import backend as K
from keras.utils import multi_gpu_model
import numpy as np
from keras.optimizers import Adadelta, SGD, Adam
from keras.backend import tf as ktf
from keras import callbacks
import argparse


parser = argparse.ArgumentParser(description='Simple CNN on ICIAR')
parser.add_argument('--res_path',
                    default='/workspace/results_keras/simple_results')
parser.add_argument('--batch_size',default=25, type=int)
parser.add_argument('--epochs',default=50, type=int)
parser.add_argument('--lr',default=0.0001, type=float)
parser.add_argument('--dropout',default=0.5, type=float)
parser.add_argument('--num_neurons',default=2048, type=int)

args = parser.parse_args()
print(args)


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

x = np.append(x_train,x_test,axis=0)
y = np.append(y_train,y_test,axis=0)
x_splits = np.array_split(x,4)
y_splits = np.array_split(y,4)




def create_model(input_shape,dropout_rate, fc_neurons,lr):
	model = Sequential()
	model.add(Lambda(lambda image: ktf.image.resize_images(image, (512,512)),
                          input_shape=input_shape))
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

        optimizer = Adam(lr=lr)


        parallel_model = multi_gpu_model(model,gpus=4)
        parallel_model.compile(optimizer=optimizer,
                               loss=keras.losses.categorical_crossentropy,
                               metrics = ['accuracy'])
        return parallel_model



for i in range(4):
	res_path = args.res_path + str(i)
	log = callbacks.CSVLogger(res_path + '/log.csv')
	tb = callbacks.TensorBoard(res_path + '/tensorboard-logs',
                           batch_size=args.batch_size,
                           write_graph=True,write_images=False,
                           histogram_freq=1)
	checkpoint = callbacks.ModelCheckpoint(res_path + '/weights-{epoch:02d}.h5',
                                       monitor='val_capsnet_acc',
                                       save_best_only=True, save_weights_only=True,
                                       verbose=1)
	lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr*(0.9**epoch))
	early_stop = callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=5)



	print('Fold ' + str(i))
	print('Creating model')
	model = create_model(input_shape=[1536,2048,3],
		dropout_rate=args.dropout, fc_neurons=args.num_neurons, lr=args.lr)
	
	model.summary()


	x_test = x_splits[i]
	y_test = y_splits[i]
	x_train = x_splits
	del x_train[i]
	y_train = y_splits
	del y_train[i]
	x_train = np.concatenate(x_train,axis=0)
	y_train = np.concatenate(y_train,axis=0)
	print(x_train.shape)
	print(y_train.shape)

	print('Fitting model')
	his = model.fit(x=x_train,y=y_train,batch_size=args.batch_size,
			epochs=args.epochs,
			verbose=1,
			validation_data=(x_test,y_test),
                        callbacks=[log, tb, early_stop, checkpoint, lr_decay])




	model.save(res_path + '/model.h5')
	del model
