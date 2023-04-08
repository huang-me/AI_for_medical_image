#!/bin/bash

python3 main.py \
	-num_epochs 150 \
	-batch_size 64 \
	-lr 0.001 \
	-dropout_rate 0.25 \
	-activation_function 'relu' \
	-elu_alpha 2.0

python3 main.py \
	-num_epochs 150 \
	-batch_size 64 \
	-lr 0.001 \
	-dropout_rate 0.5 \
	-activation_function 'relu' \
	-elu_alpha 2.0

python3 main.py \
	-num_epochs 150 \
	-batch_size 64 \
	-lr 0.001 \
	-dropout_rate 0.75 \
	-activation_function 'relu' \
	-elu_alpha 2.0

python3 main.py \
	-num_epochs 150 \
	-batch_size 64 \
	-lr 0.001 \
	-dropout_rate 0.25 \
	-activation_function 'elu' \
	-elu_alpha 2.0

python3 main.py \
	-num_epochs 150 \
	-batch_size 64 \
	-lr 0.001 \
	-dropout_rate 0.5 \
	-activation_function 'elu' \
	-elu_alpha 2.0

python3 main.py \
	-num_epochs 150 \
	-batch_size 64 \
	-lr 0.001 \
	-dropout_rate 0.75 \
	-activation_function 'elu' \
	-elu_alpha 2.0

python3 main.py \
	-num_epochs 150 \
	-batch_size 64 \
	-lr 0.001 \
	-dropout_rate 0.25 \
	-activation_function 'leakyrelu' \
	-elu_alpha 2.0

python3 main.py \
	-num_epochs 150 \
	-batch_size 64 \
	-lr 0.001 \
	-dropout_rate 0.5 \
	-activation_function 'leakyrelu' \
	-elu_alpha 2.0

python3 main.py \
	-num_epochs 150 \
	-batch_size 64 \
	-lr 0.001 \
	-dropout_rate 0.75 \
	-activation_function 'leakyrelu' \
	-elu_alpha 2.0
