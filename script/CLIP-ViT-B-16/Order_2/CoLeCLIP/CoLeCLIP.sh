#!/bin/bash
set -v
set -e
set -x

exp_no=CoLeCLIP_order2
config_name=CoLeCLIP

GPU=3
dataset=(StanfordCars Food MNIST OxfordPet Flowers SUN397 Aircraft Caltech101 DTD EuroSAT CIFAR100)
epochs=(8 1 2 18 63 1 20 10 35 3 2)


# training
# first dataset
CUDA_VISIBLE_DEVICES=${GPU} python main.py \
    --config-file configs/${config_name}.yaml \
    DATASETS.TRAIN  [\'${dataset[0]}\'] \
    SOLVER.EPOCHS ${epochs[0]} \
    METHOD.NUM_PROMPTS_PER_TASK 1 \
    METHOD.STAGE_STEP 0.5 \
    SAVE ckpt/exp_${exp_no} \
    METHOD.VOCAB_PTH None \
    METHOD.TASK_CLS_DICT None \
    SOLVER.LR 0.001 \
    METHOD.PERCENTAGE 0.7 \
    METHOD.MOM_COEF 0.1 \
    METHOD.MEM True \
    TASK_ORDER "['StanfordCars', 'Food', 'MNIST', 'OxfordPet', 'Flowers', 'SUN397', 'Aircraft', 'Caltech101', 'DTD', 'EuroSAT', 'CIFAR100']" \
    DATASETS.EVAL "['StanfordCars', 'Food', 'MNIST', 'OxfordPet', 'Flowers', 'SUN397', 'Aircraft', 'Caltech101', 'DTD', 'EuroSAT', 'CIFAR100']"

for ((i = 1; i < ${#dataset[@]}; i++)); do
    dataset_cur=${dataset[i]}
    dataset_pre=${dataset[i - 1]}

    # continue training
    CUDA_VISIBLE_DEVICES=${GPU} python main.py \
        --config-file configs/${config_name}.yaml \
        DATASETS.TRAIN [\'${dataset_cur}\'] \
        SOLVER.EPOCHS ${epochs[i]} \
        SAVE ckpt/exp_${exp_no} \
        MODEL.LOAD ckpt/exp_${exp_no}/${dataset_pre}.pth \
        METHOD.NUM_PROMPTS_PER_TASK 1 \
        METHOD.STAGE_STEP 0.5 \
        METHOD.VOCAB_PTH ckpt/exp_${exp_no}/${dataset_pre}_vocab.pth \
        METHOD.TASK_CLS_DICT ckpt/exp_${exp_no}/${dataset_pre}.json \
        SOLVER.LR 0.001 \
        METHOD.PERCENTAGE 0.7 \
        METHOD.MOM_COEF 0.1 \
        METHOD.MEM True \
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
        MODEL.LOAD ckpt/exp_${exp_no}/${dataset_cur}.pth \
        SAVE ckpt/exp_${exp_no} \
        METHOD.NUM_PROMPTS_PER_TASK 1 \
        METHOD.STAGE_STEP 0.5 \
        METHOD.VOCAB_PTH ckpt/exp_${exp_no}/${dataset_cur}_vocab.pth \
        METHOD.TASK_CLS_DICT ckpt/exp_${exp_no}/${dataset_cur}.json \
        SOLVER.LR 0.001 \
        METHOD.PERCENTAGE 0.7 \
        METHOD.MOM_COEF 0.1 \
        METHOD.MEM True \
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