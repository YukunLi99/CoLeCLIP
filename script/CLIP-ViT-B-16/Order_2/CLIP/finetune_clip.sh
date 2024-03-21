#!/bin/bash
set -v
set -e
set -x

# train fine-tune CLIP
exp_no=train_finetune_order2
config_name=Finetune

GPU=0
dataset=(StanfordCars Food MNIST OxfordPet Flowers SUN397 Aircraft Caltech101 DTD EuroSAT CIFAR100)
lr=(1e-5 1e-5 5e-5 1e-5 1e-5 1e-5 5e-5 1e-5 1e-5 1e-5 1e-5 1e-5)

# training
# first dataset
CUDA_VISIBLE_DEVICES=${GPU} python main.py \
    --config-file configs/${config_name}.yaml \
    DATASETS.TRAIN  [\'${dataset[0]}\'] \
    SOLVER.LR ${lr[i]} \
    SAVE ckpt/exp_${exp_no} \
    SOLVER.ITERATIONS 1000 \
    TASK_ORDER "['StanfordCars', 'Food', 'MNIST', 'OxfordPet', 'Flowers', 'SUN397', 'Aircraft', 'Caltech101', 'DTD', 'EuroSAT', 'CIFAR100']" \
    DATASETS.EVAL "['StanfordCars', 'Food', 'MNIST', 'OxfordPet', 'Flowers', 'SUN397', 'Aircraft', 'Caltech101', 'DTD', 'EuroSAT', 'CIFAR100']"

    
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
        TASK_ORDER "['StanfordCars', 'Food', 'MNIST', 'OxfordPet', 'Flowers', 'SUN397', 'Aircraft', 'Caltech101', 'DTD', 'EuroSAT', 'CIFAR100']" \
        DATASETS.EVAL "['StanfordCars', 'Food', 'MNIST', 'OxfordPet', 'Flowers', 'SUN397', 'Aircraft', 'Caltech101', 'DTD', 'EuroSAT', 'CIFAR100']"
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
