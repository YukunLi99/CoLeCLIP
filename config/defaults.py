# Copyright (c) Facebook, Inc. and its affiliates.
from .config import CfgNode as CN

# NOTE: given the new config system
# (https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html),
# we will stop adding new functionalities to default CfgNode.

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 2


# Model
_C.MODEL = CN()
_C.MODEL.FINETUNE_MODE = "full"  # full tuning   choices=["full", "peft"]
_C.MODEL.TRAIN_MODE = "whole"    # "whole" "text" "image"
_C.MODEL.LOAD = None
_C.MODEL.VLM = CN()
_C.MODEL.VLM.NAME = "ViT-B/16"
_C.MODEL.VLM.LOAD = "./weights/clip/ViT-B-16.pt"


#Method
_C.METHOD = CN()
_C.METHOD.NAME = "finetune"  # choice("Finetune", "ZSCL", 'LAE', 'CoLeCLIP', 'CODAPrompt', 'L2P++', 'DualPrompt')

# wc
_C.METHOD.L2 = 0

 # we_wise
_C.METHOD.WE_WISE = False
_C.METHOD.WE_WISE_ALPHA = 0.98
_C.METHOD.MOVING_AVG = False
_C.METHOD.MV_AVG_DECAY = 0.999


 # zscl
_C.METHOD.IMAGE_LOSS = False
_C.METHOD.TEXT_LOSS = False
_C.METHOD.REF_WISE_ALPHA = 0.8
_C.METHOD.REF_WISE = False
_C.METHOD.REF_MODEL = None
_C.METHOD.REF_DATASET = None
_C.METHOD.REF_SENTENCES = None
_C.METHOD.T = 2.0
_C.METHOD.NUM = 64
_C.METHOD.TEXT_DATASETS = None

 # we
_C.METHOD.WE = False
_C.METHOD.AVG_FREQ = 100

# others
_C.METHOD.WEIGHT_ADJUST = False
_C.METHOD.FEATURE_MSE = False
_C.METHOD.ABLATION_LOSS_2 = False
_C.METHOD.WISE_MERGE = False
_C.METHOD.WISE_FT = False
_C.METHOD.WISE_FT_MODEL = "n"   # choices=["n", "zeroshot"]
_C.METHOD.WISE_FT_ALPHA = 0.8
_C.METHOD.MV_AVG_MODEL = "n"    # choices=["n", "t", "zeroshot"],
_C.METHOD.ALPHA = 0.5

# lwf
_C.METHOD.LWF = False

# iCaRL
_C.METHOD.DATASET_ORDER = None
_C.METHOD.MEMORY_SIZE = 10000

# LAE
_C.METHOD.VIS = False
_C.METHOD.TXT = False
_C.METHOD.PET_CLS = None
_C.METHOD.ADAPT_BLOCKS = None

_C.METHOD.PET_KWARGS = CN()
# adapter
_C.METHOD.PET_KWARGS.down_sample = 5
_C.METHOD.PET_KWARGS.mode = "parallel"
_C.METHOD.PET_KWARGS.scale = None

#lora
_C.METHOD.PET_KWARGS.rank = 5

# prefix
_C.METHOD.PET_KWARGS.length = 10
_C.METHOD.PET_KWARGS.key_scale = None
_C.METHOD.PET_KWARGS.val_scale = None
_C.METHOD.PET_KWARGS.compensatory = True
_C.METHOD.PET_KWARGS.position = 0

_C.METHOD.NUM_EMAS = 1
_C.METHOD.EMA_DECAY = 0.9999
_C.METHOD.EVAL_ONLY_EMAS = False
_C.METHOD.EVAL_ONLY_ONLINE = False

#Promptzoo
_C.METHOD.PROMPT_PARAM = [100, 8, 0.0]  # e prompt pool size, e prompt length, g prompt length
_C.METHOD.NUM_TASKS = 11 # number of task
_C.METHOD.G_LAYER = None
_C.METHOD.E_LAYER = None




# CoLeCLIP
_C.ZERO_SHOT_WEIGHT = './weights/zero_shot_vocab/vit_b_16_zero_shot_vocab.pth'
_C.METHOD.NUM_PROMPTS_PER_TASK = 0
_C.METHOD.STAGE_STEP = None #1.0  two-stage fine-tuning


_C.METHOD.USE_VOC = False   # class vocabulary
_C.METHOD.VOCAB_PTH = None   # store class vocabulary
_C.METHOD.TASK_CLS_DICT = None  # store classes of each task


_C.METHOD.MOM_COEF = 0.5    # Momentum coefficient alpha
_C.METHOD.MEM = False         # stable momentum update

# Negative Class Label Selection
_C.METHOD.PERCENTAGE = 0.7  # percentage threshold gamma


_C.SAVE = "ckpt/exp_zscl"
_C.TEMPLATE = None


# train
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 64

# test
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 128
_C.TEST.CUR_TASK = None

# dataset
_C.DATASETS = CN()
_C.DATASETS.DATA_LOCATION = "./Image"
_C.DATASETS.TRAIN = []
_C.DATASETS.EVAL = []

# solver
_C.SOLVER = CN()
_C.SOLVER.SEED = 42

_C.SOLVER.LR = 0.001  # Learning rate
_C.SOLVER.WD = 0.0    # Weight decay
_C.SOLVER.LS = 0.0    # Label smoothing
_C.SOLVER.WARMUP_LENGTH = 100
_C.SOLVER.BETA2 = 0.999
_C.SOLVER.ITERATIONS = None
_C.SOLVER.EPOCHS = None
_C.SOLVER.LOSS_INTERVAL = 20
_C.SOLVER.EVAL_EVERY_EPOCH = False

_C.SOLVER.DEVICE = "cuda"
_C.SOLVER.CYCLE = 0
_C.SOLVER.SCALE_MIN = 0.0
_C.SOLVER.SCHEDULE_UNIT = "step"

# Task Order
_C.TASK_ORDER = ['Aircraft', 'Caltech101' , 'CIFAR100', 'DTD', 'EuroSAT', 'Flowers', 'Food', 'MNIST', 'OxfordPet', 'StanfordCars', 'SUN397']
