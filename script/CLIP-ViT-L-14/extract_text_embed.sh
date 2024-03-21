#!/bin/bash
set -v
set -e
set -x

# extract text embedding from CLIP
CUDA_VISIBLE_DEVICES=0 python main.py --eval-only \
    --extract-text \
    --config-file configs/CLIP.yaml \
    MODEL.VLM.NAME ViT-L/14 \
    MODEL.VLM.LOAD  ./weights/clip/ViT-L-14.pt \
    ZERO_SHOT_WEIGHT './weights/zero_shot_vocab/vit_l_14_zero_shot_vocab.pth'