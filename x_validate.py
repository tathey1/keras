from keras.datasets import pathology

def k_fold_xval(k, out_path, model)
	
	X,Y = pathology.load_all_data()
	num_examples = X.shape[0]
	if num_examples % k != 0
		raise ValueError('Number of folds does not fit into number of examples: ' +  str(num_examples))
	
	num_val = num_examples/k
	num_train = num_examples - num_val
	train_idx = range(num_train)
	for fold in range(k):
		x_val = X[num_val*fold:num_val*(fold+1)]
		x_train = X.take(train_idx+num_val*(fold+1),mode='wrap')
