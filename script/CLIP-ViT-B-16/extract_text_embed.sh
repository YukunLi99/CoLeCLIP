#!/bin/bash
set -v
set -e
set -x

# extract text embedding from CLIP
CUDA_VISIBLE_DEVICES=0 python main.py --eval-only \
    --extract-text \
    --config-file configs/CLIP.yaml \
    ZERO_SHOT_WEIGHT './weights/zero_shot_vocab/vit_b_16_zero_shot_vocab.pth'