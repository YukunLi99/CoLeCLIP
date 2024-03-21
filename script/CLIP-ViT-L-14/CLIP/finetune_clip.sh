#!/bin/bash
set -v
set -e
set -x

# train fine-tune CLIP
exp_no=train_finetune_vit_14_order1
config_name=Finetune

GPU=0
dataset=(Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST OxfordPet StanfordCars SUN397)
lr=(5e-5 1e-5 1e-5 1e-5 1e-5 1e-5 1e-5 5e-5 1e-5 1e-5 1e-5 1e-5)

# training
# first dataset
CUDA_VISIBLE_DEVICES=${GPU} python main.py \
    --config-file configs/${config_name}.yaml \
    DATASETS.TRAIN  [\'${dataset[0]}\'] \
    SOLVER.LR ${lr[i]} \
    SAVE ckpt/exp_${exp_no} \
    SOLVER.ITERATIONS 1000 \
    MODEL.VLM.NAME ViT-L/14 \
    MODEL.VLM.LOAD  ./weights/clip/ViT-L-14.pt

    
for ((i = 1; i < ${#dataset[@]}; i++)); do
    dataset_cur=${dataset[i]}
    dataset_pre=${dataset[i - 1]}

    # continue training
    CUDA_VISIBLE_DEVICES=${GPU} python main.py \
        --config-file configs/${config_name}.yaml \
        DATASETS.TRAIN [\'${dataset_cur}\'] \
        SOLVER.LR ${lr[i]} \
        SAVE ckpt/exp_${exp_no} \
        MODEL.LOAD ckpt/exp_${exp_no}/${dataset_pre}.pth \
        SOLVER.ITERATIONS 1000 \
        MODEL.VLM.NAME ViT-L/14 \
        MODEL.VLM.LOAD  ./weights/clip/ViT-L-14.pt
done

# eval
for ((i = 0; i < ${#dataset[@]}; i++)); do
    dataset_cur=${dataset[i]}
    CUDA_VISIBLE_DEVICES=${GPU} python main.py --eval-only \
        --config-file configs/${config_name}.yaml \
        DATASETS.TRAIN [\'${dataset_cur}\'] \
        TEST.CUR_TASK ${dataset_cur} \
        SAVE ckpt/exp_${exp_no} \
        MODEL.LOAD ckpt/exp_${exp_no}/${dataset_cur}.pth \
        MODEL.VLM.NAME ViT-L/14 \
        MODEL.VLM.LOAD  ./weights/clip/ViT-L-14.pt
done


# metric
for file in ckpt/exp_${exp_no}/*
do
    if [ "${file: -15}" == 'CIL_results.csv' ]; then
        echo $file
        python ./tools/eval_metric.py \
        --file_path $file
    elif [ "${file: -15}" == 'TIL_results.csv' ]; then
        echo $file
        python ./tools/eval_metric.py \
        --file_path $file
    fi
done
