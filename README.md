# CoLeCLIP: Open-Domain Continual Learning via Joint Task Prompt and Vocabulary Learning
[![paper](https://img.shields.io/badge/arXiv-2403.10245-<COLOR>.svg)](https://arxiv.org/abs/2403.10245)


## 📰 News
- [2024.03.21] We release the code for Open-Domain Continual Learning with both task- and class-incremental learning setting.
- [2024.03.15] Initial [arXiv](https://arxiv.org/abs/2403.10245) submission.

## 🔨 Install
Here we provide the command lines to build conda environment.
```shell
# create enviroment using Miniconda (or Anaconda)
conda create -n coleclip python=3.9.18
conda activate coleclip

# install pytorch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu116

# install other dependencies
pip install -r requirements.txt
```

## 📚 Dataset
Open-Domain Continual Learning consists of 11 datasets from diverse domains, including `Aircraft`, `Caltech101`,`CIFAR10`, `CIFAR100`, `DTD`, `EuroSAT`, `Flowers`, `Food`, `MNIST`, `OxfordPet`,`StanfordCars` and `SUN397`.

1. Create a dataset root diretory, _e.g._, `Image`.
2. Datasets such as `Aircraft`, `Caltech101`, `CIFAR10`, etc., will be automatically downloaded.  You can refer to [datasets.md](https://github.com/Thunderbeee/ZSCL/blob/main/mtil/datasets.md) for more details.

You can also refer to the following download paths for each dataset:
[Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz), [Caltech101](https://data.caltech.edu/records/mzrjq-6wc02), [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz), [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz), [EuroSAT](https://madm.dfki.de/files/sentinel/EuroSAT.zip), [Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), [Food](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz), [MNIST](http://yann.lecun.com/exdb/mnist/), [OxfordPet](https://www.robots.ox.ac.uk/~vgg/data/pets), [StanfordCars](https://ai.stanford.edu/~jkrause/car196/cars_train.tgz), [SUN397](http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz)

## 📝 Experiment
### Weights
We use the weights of CLIP as the initial weights. You should download them first and place them in the `weights/clip`.

ViT-B/16: [Official Download Link](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)

ViT-L/14: [Official Download Link](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt)

### Extract text embedding from CLIP
```
CUDA_VISIBLE_DEVICES=0 python main.py --eval-only \
    --extract-text \
    --config-file configs/CLIP.yaml \
    ZERO_SHOT_WEIGHT './weights/zero_shot_vocab/vit_b_16_zero_shot_vocab.pth'
```
You can also run script [ViT-B/16.sh](script/CLIP-ViT-B-16/extract_text_embed.sh) for ViT-B/16 or script [ViT-L/14.sh](script/CLIP-ViT-L-14/extract_text_embed.sh) for ViT-L/14.

### Training
Next, you can train CLIP for Open-Domain Continual Learning. For instance, you can choose to train CoLeCLIP with ViT-B/16 in Order1.
```
# First dataset (Aircraft)
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-file configs/CoLeCLIP.yaml \
    DATASETS.TRAIN  [\'Aircraft\'] \
    SOLVER.EPOCHS 20 \
    METHOD.NUM_PROMPTS_PER_TASK 1 \
    METHOD.STAGE_STEP 0.5 \
    SAVE ckpt/exp_CoLeCLIP_order1 \
    METHOD.VOCAB_PTH None \
    METHOD.TASK_CLS_DICT None \
    SOLVER.LR 0.001 \
    METHOD.PERCENTAGE 0.7 \
    METHOD.MOM_COEF 0.1 \
    METHOD.MEM True

# Second dataset (Caltech101)
CUDA_VISIBLE_DEVICES=0 python main.py \
        --config-file configs/CoLeCLIP.yaml \
        DATASETS.TRAIN [\'Caltech101\'] \
        SOLVER.EPOCHS 10 \
        SAVE ckpt/exp_CoLeCLIP_order1 \
        MODEL.LOAD ckpt/exp_CoLeCLIP_order1/Aircraft.pth \
        METHOD.NUM_PROMPTS_PER_TASK 1 \
        METHOD.STAGE_STEP 0.5 \
        METHOD.VOCAB_PTH ckpt/exp_CoLeCLIP_order1/Aircraft_vocab.pth \
        METHOD.TASK_CLS_DICT ckpt/exp_CoLeCLIP_order1/Aircraft.json \
        SOLVER.LR 0.001 \
        METHOD.PERCENTAGE 0.7 \
        METHOD.MOM_COEF 0.1 \
        METHOD.MEM True
```
We suggest using the provided scripts, such as script [CoLeCLIP.sh](script/CLIP-ViT-B-16/Order_1/CoLeCLIP/CoLeCLIP.sh) to train the model. You can find more methods and different task orders in the [script](script) directory.
If you want to record the output to a file, you can try the following command.
```shell
bash script/CLIP-ViT-B-16/Order_1/CoLeCLIP/CoLeCLIP.sh 2>&1 | tee -a ./log/CoLeCLIP.txt
```
Please note that you need to create a `log` directory.

### Evaluation
```
# Evaluation dataset (Aircraft)
CUDA_VISIBLE_DEVICES=0 python main.py --eval-only \
        --config-file configs/CoLeCLIP.yaml \
        DATASETS.TRAIN [\'Aircraft\'] \
        TEST.CUR_TASK Aircraft \
        MODEL.LOAD ckpt/exp_CoLeCLIP_order1/Aircraft.pth \
        SAVE ckpt/exp_CoLeCLIP_order1 \
        METHOD.NUM_PROMPTS_PER_TASK 1 \
        METHOD.STAGE_STEP 0.5 \
        METHOD.VOCAB_PTH ckpt/exp_CoLeCLIP_order1/Aircraft_vocab.pth \
        METHOD.TASK_CLS_DICT ckpt/exp_CoLeCLIP_order1/Aircraft.json \
        SOLVER.LR 0.001 \
        METHOD.PERCENTAGE 0.7 \
        METHOD.MOM_COEF 0.1 \
        METHOD.MEM True
```
The above scripts also include evaluation of the model, and the evaluation is automatically performed at the end of training.

## 👍 Acknowledgements
We would like to express our gratitude to the following open-source projects for their inspiration:
- [ZSCL](https://github.com/Thunderbeee/ZSCL)
- [CODA-Prompt](https://github.com/GT-RIPL/CODA-Prompt)
- [LAE](https://github.com/gqk/LAE)
- [CLIP](https://github.com/openai/CLIP)

## 🎫 Lincese
The content of this project itself is licensed under [LICENSE](LICENSE).

## 📇 Cite our Paper
If you find this project useful in your research, please consider cite:

```
@article{Li2024coleclip,
  title={CoLeCLIP: Open-Domain Continual Learning via Joint Task Prompt and Vocabulary Learning},
  journal={arXiv preprint arXiv:2403.10245},
  year={2024}
}
```
##   

If you like our project, please give us a star ⭐ on GitHub for latest updates!