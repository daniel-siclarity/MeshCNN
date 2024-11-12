#!/usr/bin/env bash

## run the training
# Inside scripts/shrec/train.sh
python train.py \
  --dataset_mode classification \
  --dataroot datasets/shrec_16 \
  --name shrec16 \
  --ncf 64 128 256 256 \
  --ninput_edges 750 \
  --pool_res 600 450 300 180 \
  --fc_n 100 \
  --norm group \
  --num_groups 16 \
  --resblocks 1 \
  --flip_edges 0.0 \
  --slide_verts 0.0 \
  --arch mconvnet \
  --lr 0.001 \
  --batch_size 16 \
  --num_aug 20 \
  --niter 100 \
  --niter_decay 100 \
  --epoch_count 159 \
  --continue_train \
  --which_epoch 158