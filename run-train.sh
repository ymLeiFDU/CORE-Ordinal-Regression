#!/bin/bash

gpus='3'

if [[ $# = 1 ]]; then
	gpus=${1}
fi

python main.py \
--epochs 50 \
--lr 0.0001 \
--train_batch_size 32 \
--test_batch_size 32 \
--momentum 0.9 \
--weight_decay 0.0001 \
--step_size 20 \
--schedular_gamma 0.2 \
--gpus ${gpus} \
--workers 16 \
--ckpt_interval 5 \
--dataset MORPH \
--fold 1 \
--model vgg16_bn \
--optimizer Adam \
--save_files y \
--run_tag SORD+CORE \
















