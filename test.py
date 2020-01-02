import argparse
import os
import numpy as np
import math
import itertools
from datetime import date

import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision
from config import *
from utils import *

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from data import Fashion_attr_prediction
from aae import * # import the model 

import torch.nn as nn
import torch.nn.functional as F
import torch

# import fid score evaluator 
from pytorch_fid.fid_score import *
from pytorch_fid.inception import *

cuda = True if torch.cuda.is_available() else False
today = date.today().strftime("%Y%m%d")

def test():
	device = torch.device("cuda" if cuda else "cpu")

	# load the model
	discriminator = Discriminator().to(device)
	discriminator.load_state_dict(load_model("aae_discriminator", CONFIG_AS_STR, today))
	discriminator.eval()

	if cuda:
		encoder.cuda()
		decoder.cuda()
		discriminator.cuda()
		adversarial_loss.cuda()
		pixelwise_loss.cuda()


	# generate fixed noise vector
	n_row = 10
	fixed_noise = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, LATENT_DIM))))
	name = gen_name("aae", CONFIG_AS_STR, today, "test")

	if FIXED_NOISE:
		sample_image(decoder=decoder, n_row=n_row, name=name, fixed_noise=fixed_noise)
	else:
		sample_image(decoder=decoder, n_row=n_row, name=name)


	path = ["data/Img", "images/%s/%s" % (CATEGORIES_AS_STR, name) ]
	fid_value = calculate_fid_given_paths(path,
                                          TEST_BATCH_SIZE,
                                          cuda)

	print('FID: ', fid_value)

if __name__ == '__main__':
	test()

