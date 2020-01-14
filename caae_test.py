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

def class_noise(class_num, dim, size):
	'''if class_num == CATEGORIES[0]:
		mu = 3
	else:
		mu = -3
	mean = np.ones(dim) * mu
	cov = np.diag(np.ones(dim))
	arr = np.random.multivariate_normal(mean, cov, size)'''
	l = class_num
	half = int(dim/2)
	m1 = 10*np.cos((l*2*np.pi)/10)
	m2 = 10*np.sin((l*2*np.pi)/10)
	mean = [m1, m2]
	mean = np.tile(mean, half)
	v1 = [np.cos((l*2*np.pi)/10), np.sin((l*2*np.pi)/10)]
	v2 = [-np.sin((l*2*np.pi)/10), np.cos((l*2*np.pi)/10)]
	a1 = 8
	a2 = .4
	M =np.vstack((v1,v2)).T
	S = np.array([[a1, 0], [0, a2]])
	c = np.dot(np.dot(M, S), np.linalg.inv(M))
	cov = np.zeros((dim, dim))
	for i in range(half):
		cov[i*2:(i+1)*2, i*2:(i+1)*2] = c
	#cov = cov*cov.T
	vec = np.random.multivariate_normal(mean=mean, cov=cov,
										size=size)
	return vec

def sample_noise(size):
	noise_vector = np.zeros((size, LATENT_DIM))
	'''half = int(size/2)
	noise_vector[:half,:] = class_noise(CATEGORIES[0], LATENT_DIM, half)
	noise_vector[half:,:] = class_noise(CATEGORIES[1], LATENT_DIM, size-half)'''
	section = int(size/N_CLASSES)
	for i in range(N_CLASSES):
		noise_vector[i*section:min((i+1)*section, size), :] = class_noise(i, LATENT_DIM, min(section, size-section*i))

	return noise_vector

def test_decoder(ver):
    noise = sample_noise(1000)
    plt.figure()
    for i in range(10):
        plt.scatter(noise[i*100:(i+1)*100, 0], noise[i*100:(i+1)*100,1], s=0.5, label=str(CATEGORIES[i]))
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.legend()
    plt.title("Distribution of Noise Used to Generate 10 Categories")
    plt.show()

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

def test_encoder(ver):
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
    print("got data")
    points = []
    n_classes = len(CATEGORIES)
    for i in range(n_classes):
        points.append(np.zeros((1, LATENT_DIM)))
    for i, (imgs, labels) in enumerate(dataloader):
        real_imgs = Variable(imgs.type(Tensor))
        encoded_imgs = encoder(real_imgs).cpu().data.numpy()
        for j in range(n_classes):
            #print(points[j].shape)
            #print(encoded_imgs[np.where(labels == CATEGORIES[j]),:][].shape)
            points[j] = np.append(points[j], encoded_imgs[np.where(labels == CATEGORIES[j]),:][0], axis=0)
    
    for i in range(n_classes):
        plt.figure(i)
        plt.scatter(points[i][:,0], points[i][:,1], s=0.5)
        plt.title("Encoded Images for Category {}".format(CATEGORIES[i]))
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
        plt.savefig('10Cat_{}_400.png'.format(i))

    #for j in range(5):
    plt.figure()
    for i in range(n_classes):
        plt.scatter(points[i][:,0], points[i][:,1], s=0.1, label=str(CATEGORIES[i]))
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.legend()
    plt.title("Encoded Images for 10 Categories")
    plt.savefig('10Cat_All_400.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str, default=today, help="YYYYMMDD format")
    #parser.add_argument("--type", type=str, help="model type eg. aae")
    #parser.add_argument("--model", type=str, help="generator name eg. decoder or generator")
    opt = parser.parse_args()

    test_encoder(opt.ver)
    test_decoder(opt.ver)