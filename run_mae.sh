#!/usr/bin/env bash

# python main.py --task mae --batch_size 128 --lr 1e-4 --num_epoches 200 \
#     --n_layers 6 --n_heads 4 --ff_dim 1024 --encoder_dim 512 --decoder_dim 512 \
#     --exp_name bs128_32_lr1e-4_layer6_head4_ff1024_enc512_dec512 --gpu_id 0

# python main.py --task mae --batch_size 128 --lr 1e-4 --num_epoches 200 \
#     --n_layers 6 --n_heads 4 --ff_dim 2048 --encoder_dim 512 --decoder_dim 512 \
#     --exp_name bs128_32_lr1e-4_layer6_head4_ff2048_enc512_dec512 --gpu_id 1

# python main.py --task mae --batch_size 128 --lr 1e-4 --num_epoches 200 \
#     --n_layers 6 --n_heads 8 --ff_dim 1024 --encoder_dim 512 --decoder_dim 512 \
#     --exp_name bs128_32_lr1e-4_layer6_head8_ff1024_enc512_dec512 --gpu_id 2

python main.py --task mae --batch_size 128 --lr 1e-4 --num_epoches 200 \
    --n_layers 4 --n_heads 4 --ff_dim 1024 --encoder_dim 256 --decoder_dim 256 \
    --data_augmentation \
    --exp_name bs128_32_lr1e-4_layer4_head4_ff1024_enc256_dec256 --gpu_id 3
