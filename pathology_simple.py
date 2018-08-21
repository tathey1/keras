'''
Simple CNN model for pathology classification
'''

from __future__ import print_function
import keras
from keras.datasets import pathology
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
from keras.utils import multi_gpu_model

import numpy as np

from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adadelta, SGD


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

def create_model(dropout_rate, fc_neurons, num_layers, lr):
        model = Sequential()
        #model.add(BatchNormalization(axis=3,input_shape=input_shape))
        for i in range(num_layers):
                num_filters = 2**(i+4)
                if i==0:
                        model.add(Conv2D(num_filters, kernel_size=(3,3),
                                input_shape=input_shape))
                else:
                        model.add(Conv2D(num_filters, kernel_size=(3,3)))
                model.add(BatchNormalization(axis=3))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(2,2))

        model.add(Flatten())
        model.add(Dense(fc_neurons, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_classes, activation='softmax'))

        optimizer = SGD(lr=10,decay=0.0,momentum=0.9, nesterov=True)

        parallel_model = multi_gpu_model(model,gpus=4)
        parallel_model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=optimizer, metrics = ['accuracy'])
        return parallel_model


#new
batch_size = [12]
epochs = [60]
dropout_rate = [0.5]
fc_neurons = [128]
num_layers = [5]
learn_rate = [0.01]

total_models = len(batch_size)*len(epochs)*len(dropout_rate)*len(fc_neurons)*len(num_layers)*len(learn_rate)
counter = 0

model = create_model(dropout_rate=dropout_rate[0], fc_neurons=fc_neurons[0],
                     num_layers=num_layers[0], lr=learn_rate[0])
his = model.fit(x=x_train,y=y_train,
                batch_size=batch_size[0], epochs=epochs[0], verbose=1,
                validation_data=(x_test,y_test))
print('Train accuracy****************************************')
print(his.history['acc'])
print('Validation accuracy************************************')
print(his.history['val_acc'])
#
'''

with open('/workspace/results_keras/gridsearch.txt','w') as f:
  f.write('[Loss, accuracy]')
  f.write('\n')

  combos = [(bs,e,lr,d,dr,fn,nl) for bs in batch_size for e in epochs for lr in learn_rate for d in decay for dr in dropout_rate for fn in fc_neurons for nl in num_layers]
  for (bs,e,lr,d,dr,fn,nl) in combos:
    counter+=1
    print('Iteration %i of %i' % (counter, total_models))
    print('Batch size=%i, epochs=%i, learn_rate=%f, decay=%f, dropout_rate=%f, fc_neurons=%f, num_layers=%i' % (bs, e, lr, d, dr, fn, nl))
    f.write('Batch size=%i, epochs=%i, learn_rate=%f, decay=%f, dropout_rate=%f, fc_neurons=%f, num_layers=%i' % (bs, e, lr, d, dr, fn, nl))
    f.write('\n')
    model = create_model(dr, fn, nl, lr, d)
    model.fit(x=x_train, y=y_train, batch_size=bs, epochs=e, verbose=1)
    txt = model.evaluate(x=x_test, y=y_test, batch_size=bs)
    print(txt)
    f.write(str(txt))
    f.write('\n')
'''
