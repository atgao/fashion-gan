# -*- coding: utf-8 -*-
import torchvision.transforms as transforms

# deprecated
# EXTRACT_BATCH_SIZE = 128
# TEST_BATCH_COUNT = 30
# NUM_WORKERS = 4
# LOG_INTERVAL = 10
# DUMP_INTERVAL = 500
# TEST_INTERVAL = 100
# MOMENTUM = 0.5
# DUMPED_MODEL = "model_10_final.pth.tar"
# ENABLE_INSHOP_DATASET = False
# INSHOP_DATASET_PRECENT = 0.8
INTER_DIM_1 = 1000
INTER_DIM_2 = 1024

TEST_BATCH_SIZE = 32
FIXED_NOISE = True

TRAIN_BATCH_SIZE = 128
LR = 0.001
N_EPOCHS = 500
SAMPLE_INTERVAL = 200
NUM_WORKERS = 4
DATASET_BASE = 'data/'
GENERATED_BASE = 'images/'
IMG_SIZE = 64
CROP_SIZE = 64
IMG_CHANNELS = 3
CATEGORIES = [2, 18]
N_CLASSES = 2
CATEGORIES_AS_STR = ",".join([str(c) for c in CATEGORIES])
LATENT_DIM = 20
N_CRITIC = 1
CONFIG_AS_STR = "_".join([str(c) for c in [CATEGORIES_AS_STR, LATENT_DIM, IMG_SIZE, N_EPOCHS, LR, TRAIN_BATCH_SIZE, N_CRITIC]])
# DISTANCE_METRIC = ('euclidean', 'euclidean')

TRANSFORM_FN = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(CROP_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

TEST_TRANSFORM_FN = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])