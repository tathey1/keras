import os
from x_validate import k_fold_xval
from my_models import simple

batch_sizes = [30]
epochss = [10]
lrs = [0.0001]
dropouts = [0.2]
fc_neuronss = [2048]

num_filters = [64, 128, 256, 256, 256]
conv_kernels = [4, 4, 6, 6, 8]
strides = [2, 2, 2, 2, 2]

k=4
num_classes=4
input_shape=[1536, 2048,3]
folder = '/workspace/results_keras/simple/hyperparam/'


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
		'strides':strides, 'input_shape':input_shape, 'num_classes':num_classes}
	model = simple.simple(args_dict)

	print(str(k) + ' fold cross validation')
	args_dict = {'batch_size_train':batch_size, 'batch_size_val':10,
		'epochs':epochs}
	acc = k_fold_xval(k, combo_path, model,args_dict)

	print('Cumulative accuracy:')
	print(acc)
	f.write(str(acc) + '\n')
	
f.close()
