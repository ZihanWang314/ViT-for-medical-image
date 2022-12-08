#!/usr/bin/env bash

python main.py --task reg --batch_size 128 --lr 1e-4 --num_epoches 400 \
    --n_layers 6 --n_heads 4 --ff_dim 2048 --encoder_dim 512 --decoder_dim 512 \
    --exp_name regression --gpu_id 2 #--load_pretraining
