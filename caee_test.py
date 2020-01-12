import argparse
import os
import numpy as np
import math
import itertools
import pprint
from datetime import date
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision
from config import *
from utils import *

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from data import Fashion_attr_prediction, GeneratedDataset
from caae import *  # import the model

import torch.nn as nn
import torch.nn.functional as F
import torch

cuda = True if torch.cuda.is_available() else False
today = date.today().strftime("%Y%m%d")

def _get_base_dataloader(sample=False):
    if sample:
        _type = "sample"
    else:
        _type = "test"

    return torch.utils.data.DataLoader(
        Fashion_attr_prediction(
            categories=CATEGORIES,
            type=_type,
            transform=TEST_TRANSFORM_FN,
            crop=True,
        ),
        batch_size=128,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )

def test_dist(ver):
    device = torch.device("cuda" if cuda else "cpu")

    # load the model
    encoder = Encoder().to(device)
    encoder.load_state_dict(load_model("caae_encoder", CONFIG_AS_STR, ver, device))
    encoder.eval()

    if cuda:
        encoder.cuda()

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    sample = False
    dataloader = _get_base_dataloader(sample=sample)

    Xs = []
    Ys = []
    for i in range(len(CATEGORIES)):
        Xs.append(np.zeros(1))
        Ys.append(np.zeros(1))
    for i, (imgs, labels) in enumerate(dataloader):
        real_imgs = Variable(imgs.type(Tensor))
        encoded_imgs = encoder(real_imgs).cpu().data.numpy()
        for i in range(len(CATEGORIES)):
            Xs[i] = np.append(Xs[i], encoded_imgs[np.where(labels == CATEGORIES[i]),0])
            Ys[i] = np.append(Ys[i], encoded_imgs[np.where(labels == CATEGORIES[i]),1])
    
    for i in range(len(CATEGORIES)):
        plt.figure(i)
        plt.scatter(Xs[i], Ys[i], s=1)
        plt.title("Distribution in Latent Code for Category {}".format(CATEGORIES[i]))
        plt.xlim(-7, 7)
        plt.ylim(-7, 7)
    plt.figure(len(CATEGORIES) + 1)
    for i in range(len(CATEGORIES)):
        plt.scatter(Xs[i], Ys[i], s=1, label=str(CATEGORIES[i]))
    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    plt.legend()
    plt.title("Distribution in Latent Code")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str, default=today, help="YYYYMMDD format")
    #parser.add_argument("--type", type=str, help="model type eg. aae")
    #parser.add_argument("--model", type=str, help="generator name eg. decoder or generator")
    opt = parser.parse_args()

    test_dist(opt.ver)