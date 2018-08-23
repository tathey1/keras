batch_sizes=(30 5)
epochs=100
lrs=(0.0001 0.001)
dropouts=(0.2 0.5)
num_neuronss=(4096 2048 1024)
for batch_size in $(seq 1 2)
do
	for lr in $(seq 1 2)
	do
		for dropout in $(seq 1 2)
		do
			for num_neurons in $(seq 1 3)
			do
				folder=/workspace/results_keras/simple/hyper_param_xval/${batch_size}${lr}${dropout}${num_neurons}/
				mkdir $folder
				CUDA_VISIBLE_DEVICES=0,1,2,3 python pathology_caps_simple.py \
				--res_path=$folder \
				--batch_size=${batch_sizes[$batch_size]} \
				--epochs=100 \
				--lr=${lrs[$lr]} \
				--dropout=${dropouts[$dropout]} \
				--num_neurons=${num_neuronss[$num_neurons]}
				
			done
		done
	done
done
