#!/bin/bash

source ../conviformer/bin/activate 

echo $CUDA_VISIBLE_DEVICES
export n_gpus=1

model_name=$1
input_size=$2
data_set=$3
resume=$4
batch_size=$5

num_epochs=300

echo num_classes = ${num_classes}
echo batch_size = ${batch_size}
echo num_epochs = ${num_epochs}
echo model_name = ${model_name}
echo n_gpus = ${n_gpus}
now=$(date +"%m-%d-%y-%H-%M")

python main_hard_hiergenus.py --eval \
					--model=${model_name} --data-set ${data_set} --input-size ${input_size} \
					--num_workers 3 --batch-size ${batch_size} --resume ${resume} \
					--output_dir ../out_conviformer/${model_name}/ --seed $p