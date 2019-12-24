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
from aae import * # import the model 

import torch.nn as nn
import torch.nn.functional as F
import torch

# ----------
#  Training
# ----------

def train(b1, b2):
	img_shape = (IMG_CHANNELS, IMG_SIZE, IMG_SIZE)

	cuda = True if torch.cuda.is_available() else False

	# dataset
	data_transform_train = transforms.Compose([
	    transforms.RandomResizedCrop(CROP_SIZE),
	    transforms.RandomHorizontalFlip(),
	    transforms.ToTensor(),
	    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	    ])

	# Use binary cross-entropy loss
	adversarial_loss = torch.nn.BCELoss()
	pixelwise_loss = torch.nn.L1Loss()

	# Initialize generator and discriminator
	encoder = Encoder()
	decoder = Decoder()
	discriminator = Discriminator()

	if cuda:
	    encoder.cuda()
	    decoder.cuda()
	    discriminator.cuda()
	    adversarial_loss.cuda()
	    pixelwise_loss.cuda()

	# Configure data loader
	os.makedirs("../../data/deepfashion", exist_ok=True)
	dataloader = torch.utils.data.DataLoader(
	    Fashion_attr_prediction(
	        type="train", 
	        transform=data_transform_train
	    ),
	    batch_size=TRAIN_BATCH_SIZE,
	    shuffle=True,
	)

	# Optimizers
	optimizer_G = torch.optim.Adam(
	    itertools.chain(encoder.parameters(), decoder.parameters()), lr=LR, betas=(b1, b2)
	)
	optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(b1, b2))

	Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

	for epoch in range(N_EPOCHS):
	    for i, (imgs, _) in enumerate(dataloader):

	        # Adversarial ground truths
	        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
	        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

	        # Configure input
	        real_imgs = Variable(imgs.type(Tensor))
	        # -----------------
	        #  Train Generator
	        # -----------------

	        optimizer_G.zero_grad()

	        encoded_imgs = encoder(real_imgs)
	        decoded_imgs = decoder(encoded_imgs)

	        # Loss measures generator's ability to fool the discriminator
	        g_loss = 0.001 * adversarial_loss(discriminator(encoded_imgs), valid) + 0.999 * pixelwise_loss(
	            decoded_imgs, real_imgs
	        )

	        g_loss.backward()
	        optimizer_G.step()

	        # ---------------------
	        #  Train Discriminator
	        # ---------------------

	        optimizer_D.zero_grad()

	        # Sample noise as discriminator ground truth
	        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], LATENT_DIM))))

	        # Measure discriminator's ability to classify real from generated samples
	        real_loss = adversarial_loss(discriminator(z), valid)
	        fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
	        d_loss = 0.5 * (real_loss + fake_loss)

	        d_loss.backward()
	        optimizer_D.step()

	        print(
	            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
	            % (epoch, N_EPOCHS, i, len(dataloader), d_loss.item(), g_loss.item())
	        )

	        batches_done = epoch * len(dataloader) + i
	        if batches_done % SAMPLE_INTERVAL == 0:
	            sample_image(decoder=decoder, n_row=10, batches_done=batches_done)
	            save_model(encoder, epoch, "encoder")
	            save_model(decoder, epoch, "decoder")
	            save_model(discriminator, epoch, "discriminator")

	return encoder, decoder, discriminator



if __name__=="__main__":
	os.makedirs("images", exist_ok=True)
	parser = argparse.ArgumentParser()
	parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
	parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
	parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
	opt = parser.parse_args()
	print(opt)

	encoder, decoder, discriminator = train(opt.b1, opt.b2)
	# ----------
	#  Save Model and create Training Log
	# ----------
	# TODO: save this to a folder logs
	print(opt)
	print("Saved Encoder to {}".format(save_model(encoder, N_EPOCHS, "encoder")))
	print("Saved Decoder to {}".format(save_model(decoder, N_EPOCHS, "decoder")))
	print("Saved Discriminator to {}".format(save_model(discriminator, N_EPOCHS, "discriminator")))