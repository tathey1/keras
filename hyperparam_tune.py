'''
Thomas Athey 9/7/18

The purpose of this script is to test different combinations of hyperparameters by performing k fold cross validation
It creates an instance of a model object (that must implement the function create_model
then it passes this object to either validate_augment.eval or x_validate.k_fold_xval in order to get overall accuracy

the output file is called "summary.txt" and has the overall average accuracies for each combination of hyperparameters
for each combination of hyperparameters it creates a folder in which the results of the k fold cross validation for
those hyperparameters will be shown
'''

import os
import x_validate
import validate_augment
from my_models import simple, max_pool, pretrained_inception_v2

#hyperparameter lists
batch_sizes = [4]
epochss = [100]
lrs = [0.000001]
dropouts = [0.5]
fc_neuronss = [256]

#specs for the model (look at the model in my_models in order to see what these are used for
num_filters = [64, 128, 256, 256, 256]
conv_kernels = [4, 4, 6, 6, 8]
strides = [1, 1, 1, 1, 1]

#other parameters that need to be attended to
gpus=2
k=10
num_classes=4
input_shape=[1536, 2048,3]

#parent directory of all the output files
folder = '/workspace/results_keras/simple/test/'

if len(os.listdir(folder)) > 0:
	raise ValueError('Folder is not empty')

total_models = len(batch_sizes)*len(epochss)*len(lrs)*len(dropouts)* \
	len(fc_neuronss)
counter=0

combos = [(batch_size, epochs, lr, dropout, fc_neurons) \
	for batch_size in batch_sizes \
	for epochs in epochss \
	for lr in lrs \
	for dropout in dropouts \
	for fc_neurons in fc_neuronss]

summary_path = os.path.join(folder,'summary.txt')
print('Results summary in ' + summary_path)
f = open(summary_path,'w')

for (batch_size, epochs, lr, dropout, fc_neurons) in combos:
	
	counter+=1
	print('Iteration %i of %i' % (counter, total_models))
	combo_path = os.path.join(folder,str(counter))
	params = 'Batch size=%i, epochs=%i, learning rate=%f, dropout rate=%f, number neurons=%i' % (batch_size, epochs, lr, dropout, fc_neurons)
	print(params)
	print('Results located in ' + combo_path)
	f.write(params + '\n')
	f.write(combo_path + '\n')
	os.makedirs(combo_path)

	args_dict = {'dropout_rate':dropout, 'fc_neurons':fc_neurons, 
		'lr':lr, 'num_filters':num_filters, 'conv_kernels':conv_kernels,
		'strides':strides, 'input_shape':input_shape, 
		'num_classes':num_classes, 'batch_size':batch_size,'gpus':gpus}
	model = max_pool.max_pool(args_dict)

	print(str(k) + ' fold cross validation')
	args_dict = {'batch_size':batch_size,
		'epochs':epochs}
	acc = x_validate.k_fold_xval(k,combo_path, model,args_dict)

	print('Cumulative accuracy:')
	print(acc)
	f.write(str(acc) + '\n')
	f.flush()

f.close()
