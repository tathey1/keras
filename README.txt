I will identify the files that are worth taking note of and how to run them
For questions about the mechanics of these files, consult the comments in the files

hyper parameter tuning:

command:
CUDA_VISIBLE_DEVICES=0,1 python hyperparam_tune.py
	calls upon:
		my_models/max_pool.py 
			this file has a function create_model which creates a keras model
			other models in my_models/ can be used as well
		x_validate.py
			reads data and trains the model with k fold cross validation
CUDA_VISIBLE_DEVICES=0,1 python pretrained.py
	calls upon:
		my_models/pretrained_inception_v2.py
		validate_augment.py
			only trains/validates on a single fold in order to use builtin data augmentation strategies
			
