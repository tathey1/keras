import keras
from keras.models import Sequential
from keras.datasets import pathology
from keras.layers import Dense, Dropout, Flatten, Conv2D, Lambda
from keras import backend as K
from keras.utils import multi_gpu_model
import numpy as np
from keras.optimizers import Adadelta, SGD, Adam
from keras.backend import tf as ktf
from keras import callbacks
import argparse



parser = argparse.ArgumentParser(description='4 fold x validation on ICIAR')
parser.add_argument('--res_path',
		default='/workspace/results_keras/simple_results')
parser.add_argument('--batch_size', default=25, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--lr', default=0.0001, type=float)

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

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

X = np.append(x_train,x_test,axis=0)
Y = np.append(y_train,y_test,axis=0)
print(X.shape)
print(Y.shape)


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
	model.compile(optimizer=optimizer,
                                loss=keras.losses.categorical_crossentropy,
                                metrics = ['accuracy'])
	return model

x_t=[]
y_t=[]
x_v=[]
y_v=[]

x_v.append(X[:100])
y_v.append(Y[:100])
x_t.append(X[100:])
y_t.append(Y[100:])

x_v.append(X[100:200])
y_v.append(Y[100:200])
x_t.append(np.append(X[:100],X[200:],axis=0))
y_t.append(np.append(Y[:100],Y[200:],axis=0))

x_v.append(X[200:300])
y_v.append(Y[200:300])
x_t.append(np.append(X[:200],X[300:],axis=0))
y_t.append(np.append(Y[:200],Y[300:],axis=0))

x_v.append(X[300:])
y_v.append(Y[300:])
x_t.append(X[:300])
y_t.append(Y[:300])

for counter in range(4):
	res_path = args.res_path + str(counter)
	log = callbacks.CSVLogger(res_path + '/log.csv')
	tb = callbacks.TensorBoard(res_path + '/tensorboard-logs',
				batch_size=args.batch_size)
	model = create_model(input_shape=[1536,2048,3],
		dropout_rate=args.dropout, fc_neurons=args.num_neurons, lr=args.lr)

	model.fit(x_t[counter], y_t[counter], batch_size=args.batch_size,
		epochs=args.epochs, verbose=1,
		validation_data=(x_v[counter], y_v[counter]), 
		callbacks=[log, tb])
	print('*****************Truth***********************')
	print(y_v[counter])
	print('******************Prediction**************')
	print(model.predict(x_v[counter], batch_size=25, verbose=1))
	model.save(res_path + '/model.h5')
