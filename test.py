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


def _get_comp_dataloader():
    return torch.utils.data.DataLoader(
        GeneratedDataset(
            base_dir="./{}/{}".format(GENERATED_BASE, CATEGORIES_AS_STR),
            transform=TEST_TRANSFORM_FN,
        ),
        batch_size=128,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )


def test(ver):
    device = torch.device("cuda" if cuda else "cpu")

    # load the model
    decoder = Decoder().to(device)
    decoder.load_state_dict(load_model("aae_decoder", CONFIG_AS_STR, ver, device))
    decoder.eval()

    if cuda:
        encoder.cuda()
        decoder.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        pixelwise_loss.cuda()

    # generate fixed noise vector
    n_row = 10
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    fixed_noise = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, LATENT_DIM))))
    name = gen_name("aae", CONFIG_AS_STR, today, "test")
    os.makedirs("%s/%s" % (GENERATED_BASE, CATEGORIES_AS_STR), exist_ok=True)

    if FIXED_NOISE:
        sample_image(decoder=decoder, n_row=n_row, name=name, fixed_noise=fixed_noise, individual=True)
    else:
        sample_image(decoder=decoder, n_row=n_row, name=name, individual=True)

    sample = True
    base_dataloader = _get_base_dataloader(sample=sample)
    comparison_dataloader = _get_comp_dataloader()
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
    opt = parser.parse_args()

    test(opt.ver)
