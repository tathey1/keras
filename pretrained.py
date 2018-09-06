'''
Thomas Athey 9/6/18

I do not recommend using this because I never got pretrained models to perform well in keras

however you can try to run this if you want
'''

import keras

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import models
from keras import layers
from keras import optimizers

from keras.backend import tf as ktf

from keras.datasets import pathology

from my_models import pretrained_inception_v2 
import validate_augment
import os

num_classes=4
fc_neurons=256
dropout_rate=0.2
lr=0.0001
batch_size=20

gpus=2
input_shape=[1536, 2048, 3]
ks=[10]
epochs=63

folder='/workspace/results_keras/simple/test/'

args_model = {'dropout_rate' : dropout_rate, 'fc_neurons': fc_neurons,
	'lr':lr, 'gpus':gpus, 'num_classes':num_classes,
	'batch_size': batch_size, 'input_shape':input_shape}



args_eval = {'batch_size':batch_size, 'epochs':epochs}

summary_path = os.path.join(folder,'summary.txt')
f = open(summary_path,'w')

for k in ks:
	fold_path = os.path.join(folder,str(k))
	print(str(k) + ' folds')
	f.write(str(k) + '\n')
	os.makedirs(fold_path)

	model = pretrained_inception_v2.pretrained_inception_v2(args_model)
	
	acc = validate_augment.eval(fold_path, model, args_eval)
	print('Cumulative accuracy')
	print(acc)

	
	f.write(str(acc) + '\n')
	f.flush()
f.close()
