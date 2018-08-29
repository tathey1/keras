import keras

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import models
from keras import layers
from keras import optimizers

from keras.backend import tf as ktf

from keras.datasets import pathology

num_classes=4
fc_neurons=256
dropout=0.2
lr=0.0001

input_shape=[1536, 2048, 3]

folder='/workspace/results_keras/pretrain/inception_v2/'

args = {'batch_size': 20, 'epochs':100}

resizer = models.Sequential()
resizer.add(layers.Lambda(lambda image: ktf.image.resize_images(image,
		(299,299)), input_shape=input_shape))


feature_extractor = InceptionResNetV2(include_top=False, input_shape=(299,299,3))
for layer in feature_extractor.layers:
	layer.trainable = False

classifier = models.Sequential()
classifier.add(layers.Flatten())
classifier.add(layers.Dense(fc_neurons,activation='relu'))
classifier.add(layers.Dropout(dropout))
classifier.add(layers.Dense(num_classes,activation='softmax'))


complete_model = models.Sequential([resizer, feature_extractor, classifier])

complete_model.summary()

optimizer = optimizers.Adam(lr=lr)

complete_model.compile(optimizer=optimizer,
	loss=keras.losses.categorical_crossentropy,
	metrics=['accuracy'])

#k_fold_xval(4, folder, classifier, 
