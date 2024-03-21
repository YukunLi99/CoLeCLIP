#!/bin/bash
set -v
set -e
set -x

# eval zero-shot CLIP
exp_no=zero_shot_order2
config_name=CLIP

GPU=3
dataset=(StanfordCars Food MNIST OxfordPet Flowers SUN397 Aircraft Caltech101 DTD EuroSAT CIFAR100)

# eval
for ((i = 0; i < ${#dataset[@]}; i++)); do
    dataset_cur=${dataset[i]}
    CUDA_VISIBLE_DEVICES=${GPU} python main.py --eval-only \
        --config-file configs/${config_name}.yaml \
        DATASETS.TRAIN [\'${dataset_cur}\'] \
        TEST.CUR_TASK ${dataset_cur} \
        SAVE ckpt/exp_${exp_no} \
        TASK_ORDER "['StanfordCars', 'Food', 'MNIST', 'OxfordPet', 'Flowers', 'SUN397', 'Aircraft', 'Caltech101', 'DTD', 'EuroSAT', 'CIFAR100']" \
        DATASETS.EVAL "['StanfordCars', 'Food', 'MNIST', 'OxfordPet', 'Flowers', 'SUN397', 'Aircraft', 'Caltech101', 'DTD', 'EuroSAT', 'CIFAR100']"
done


# metric
for file in ckpt/exp_${exp_no}/*
do
    if [ "${file: -15}" == 'CIL_results.csv' ]; then
        echo $file
        python ./tools/eval_metric.py \
        --file_path $file \
        --Order 1
    elif [ "${file: -15}" == 'TIL_results.csv' ]; then
        echo $file
        python ./tools/eval_metric.py \
        --file_path $file \
        --Order 1
    fi
done
