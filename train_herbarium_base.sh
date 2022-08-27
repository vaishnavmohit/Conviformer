#!/bin/bash

source ../conviformer/bin/activate 

echo $CUDA_VISIBLE_DEVICES

model_name='convit_base_patch' #'convit_base'
dataset=$1 # 'Herbarium22', 'Herbarium', 'INAT19'
batch_size=$2 
export n_gpus=$3 

num_epochs=300

echo num_classes = ${num_classes}
echo batch_size = ${batch_size}
echo num_epochs = ${num_epochs}
echo model_name = ${model_name}
echo n_gpus = ${n_gpus}
now=$(date +"%m-%d-%y-%H-%M")

python -m torch.distributed.launch --nproc_per_node=${n_gpus} --use_env main.py \
					--model=${model_name} --data-set ${dataset} --input-size 224 \
					--num_workers 6 --batch-size ${batch_size} --epochs ${num_epochs} \
					--output_dir ../out_conviformer/herbarium_22/${model_name}/224/ce_${now} \
					--drop ${4:-.1}