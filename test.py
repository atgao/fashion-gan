# Do not delete this
import argparse
import os
import numpy as np
import math
import itertools
import pprint
from datetime import date

import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision
from config import *
from utils import *

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from data import Fashion_attr_prediction, GeneratedDataset
from aae import *  # import the model

import torch.nn as nn
import torch.nn.functional as F
import torch

# import fid score evaluator 
from pytorch_fid.fid_score import *
from pytorch_fid.inception import *

cuda = False#True if torch.cuda.is_available() else False
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


def _get_comp_dataloader(path):
    return torch.utils.data.DataLoader(
        GeneratedDataset(
            base_dir="./{}".format(path),
            transform=TEST_TRANSFORM_FN,
        ),
        batch_size=128,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )


def test(ver, model_type, generator):
    device = torch.device("cuda" if cuda else "cpu")

    # load the model
    decoder = Decoder().to(device)
    decoder.load_state_dict(load_model(model_type + "_" + generator, CONFIG_AS_STR, ver, device))
    decoder.eval()

    if cuda:
        decoder.cuda()
    # generate fixed noise vector
    n_row = 10
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    fixed_noise = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, LATENT_DIM))))
    path = "/".join([str(c) for c in [GENERATED_BASE, model_type, CONFIG_AS_STR, "test"]])
    name = today
    os.makedirs(path, exist_ok=True)

    if FIXED_NOISE:
        sample_image(decoder=decoder, n_row=n_row, path=path, name=name, fixed_noise=fixed_noise, individual=True)
    else:
        sample_image(decoder=decoder, n_row=n_row, path=path, name=name, individual=True)

    sample = True
    base_dataloader = _get_base_dataloader(sample=sample)
    comparison_dataloader = _get_comp_dataloader(path)
    print("Calculating FID")
    if sample:
        print(f"Using sampled list")
        for category, sampled_test_images in base_dataloader.dataset.sample_dict.items():
            print(f"Category {category}: {len(sampled_test_images)} images sampled")
            pprint.pprint(sampled_test_images)
    else:
        print("Using full test list")

    fid_value = calculate_fid_given_dataloaders(base_dataloader,
                                                comparison_dataloader,
                                                cuda,
                                                2048)

    print('FID: ', fid_value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str, default=today, help="YYYYMMDD format")
    parser.add_argument("--type", type=str, default="aae", help="model type eg. aae")
    parser.add_argument("--model", type=str, default="decoder", help="generator name eg. decoder or generator")
    opt = parser.parse_args()

    test(opt.ver, opt.type, opt.model)
