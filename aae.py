from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image
from config import *
from utils import *

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from data import Fashion_attr_prediction

import torch.nn as nn
import torch.nn.functional as F
import torch



# data_transform_test = transforms.Compose([
#     transforms.RandomResizedCrop(CROP_SIZE),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

cuda = True if torch.cuda.is_available() else False

def reparameterization(mu, logvar):
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), LATENT_DIM))))
    z = sampled_z * std + mu
    return z


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(IMG_CHANNELS*IMG_SIZE**2), INTER_DIM_1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(INTER_DIM_1, INTER_DIM_1),
            nn.BatchNorm1d(INTER_DIM_1),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Linear(int(IMG_CHANNELS*IMG_SIZE**2), INTER_DIM_2),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Linear(INTER_DIM_2, INTER_DIM_1),
            #nn.BatchNorm1d(INTER_DIM_1),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Linear(INTER_DIM_1, INTER_DIM_1),
            #nn.BatchNorm1d(INTER_DIM_1),
            #nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512, LATENT_DIM)
        self.logvar = nn.Linear(512, LATENT_DIM)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(IMG_CHANNELS*IMG_SIZE**2)),
            nn.Tanh(),
            #nn.Linear(LATENT_DIM, 512),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Linear(512, 512),
            #nn.BatchNorm1d(512),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Linear(512, 1024),
            #nn.BatchNorm1d(1024),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Linear(1024, int(IMG_CHANNELS*IMG_SIZE**2)),
            #nn.Tanh(),
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *(IMG_CHANNELS, IMG_SIZE, IMG_SIZE))
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity

def sample_image(decoder, n_row, path, name, fixed_noise=None, individual=False):
    """Saves a grid of generated digits"""
    # Sample noise
    if fixed_noise is not None:
        z = fixed_noise
    else:
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        z = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, LATENT_DIM))))
    gen_imgs = decoder(z)

    if individual:
        for i in range(gen_imgs.size(0)): # save them one by one
            save_image(gen_imgs.data[i, :, :, :], "%s/%s_%d.png" % (path, name, i), normalize=True)
    else: # create grid
        save_image(gen_imgs.data, "%s/%s.png" % (path, name), nrow=n_row, normalize=True)


#def sample_image_fixed(decoder, fixed_noise, n_row, name):
#    """Saves a grid of generated digits"""
#    gen_imgs = decoder(fixed_noise)
#    save_image(gen_imgs.data, "images/%s.png" % name, nrow=n_row, normalize=True)





