'''
Simple CNN model for pathology classification
'''

from __future__ import print_function
import keras
from keras.datasets import pathology
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

batch_size = 4
num_classes = 4
epochs = 20

# input image dimensions
img_rows, img_cols = 1536, 2048

# the data, split between train and test sets
print('Loading data...')
(x_train, y_train), (x_test, y_test), label_to_class = pathology.load_data()

print('Reformatting data...')
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

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

X = np.concatenate([x_train, x_test],axis=0)
Y = np.concatenate([y_train, y_test], axis=0)

def create_model():
	model = Sequential()
	model.add(Conv2D(32,kernel_size=(3,3),
		activation='relu',
		input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(64,kernel_size=(3,3),
		activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(128,kernel_size=(3,3),
		activation='relu'))
	model.add(MaxPooling2D(pool_size=(4,4)))
	model.add(Conv2D(64,kernel_size=(3,3), strides=(2,2),
		activation='relu'))
	model.add(MaxPooling2D(pool_size=(4,4)))
	model.add(Conv2D(64,kernel_size=(3,3)))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	
	model.compile(loss=keras.losses.categorical_crossentropy,
		optimizer='adam', metrics = ['accuracy'])
	return model

model = KerasClassifier(build_fn = create_model, verbose=0)

batch_size = [4]
epochs = [1]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X,Y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

