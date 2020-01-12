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
from caae import * # import the model 

import torch.nn as nn
import torch.nn.functional as F
import torch

# ----------
#  Training
# ----------
cuda = True if torch.cuda.is_available() else False
today = date.today().strftime("%Y%m%d")
def class_noise(class_num, dim, size):
	if class_num == CATEGORIES[0]:
		mu = 3
	else:
		mu = -3
	#print(dim)
	mean = np.ones(dim) * mu
	#print(mean)
	cov = np.diag(np.ones(dim))
	arr = np.random.multivariate_normal(mean, cov, size)
	return arr

def sample_noise(size):
	noise_vector = np.zeros((size, LATENT_DIM))
	half = int(size/2)
	noise_vector[:half,:] = class_noise(CATEGORIES[0], LATENT_DIM, half)
	noise_vector[half:,:] = class_noise(CATEGORIES[1], LATENT_DIM, size-half)

	return noise_vector

def one_hot_encode(index1, index2):
	arr = np.zeros((len(index1) + len(index2), N_CLASSES))
	arr[index1, 0] = 1
	arr[index2, 1] = 1
	return arr

def train(b1, b2):
	# Use binary cross-entropy loss
	adversarial_loss = torch.nn.BCELoss()
	pixelwise_loss = torch.nn.L1Loss()

	device = torch.device("cuda" if cuda else "cpu")
	# Initialize generator and discriminator
	encoder = Encoder().to(device)
	decoder = Decoder().to(device)
	discriminator = Discriminator().to(device)

	if cuda:
		encoder.cuda()
		decoder.cuda()
		discriminator.cuda()
		adversarial_loss.cuda()
		pixelwise_loss.cuda()

	# Configure data loader
	# os.makedirs("../../data/deepfashion", exist_ok=True)
	dataloader = torch.utils.data.DataLoader(
		Fashion_attr_prediction(
            categories=CATEGORIES,
			type="train", 
			transform=TRANSFORM_FN,
			crop=True
		),
		batch_size=TRAIN_BATCH_SIZE,
		num_workers=NUM_WORKERS,
		shuffle=True,
	)

	# Optimizers
	optimizer_G = torch.optim.Adam(
		itertools.chain(encoder.parameters(), decoder.parameters()), lr=LR, betas=(b1, b2)
	)
	optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(b1, b2))

	Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

	# generate fixed noise vector
	n_row = 10
	#fixed_noise = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, LATENT_DIM))))
	#noise_vector = np.zeros((n_row**2, LATENT_DIM))
	#noise_vector[:50,:] = class_noise(CATEGORIES[0], LATENT_DIM, 50)
	#noise_vector[50:,:] = class_noise(CATEGORIES[1], LATENT_DIM, 50)
	fixed_noise = Variable(Tensor(sample_noise(n_row**2)))
	# make directory for saving images
	path = "/".join([str(c) for c in [GENERATED_BASE, "caae", CONFIG_AS_STR, "train"]])
	os.makedirs(path, exist_ok=True)

	# save losses across all
	G_losses = []
	D_losses = []

	one_hot_label = one_hot_encode(range(int(TRAIN_BATCH_SIZE/2)), range(int(TRAIN_BATCH_SIZE/2), TRAIN_BATCH_SIZE))

	# training loop 
	for epoch in range(N_EPOCHS):
		for i, (imgs, labels) in enumerate(dataloader):
			# Configure input
			real_imgs = Variable(imgs.type(Tensor))

			# Adversarial ground truths
			valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
			fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

			# ---------------------
			#  Train Discriminator
			# ---------------------

			optimizer_D.zero_grad()

			# Sample noise as discriminator ground truth
			#z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], LATENT_DIM))))
			z = Variable(Tensor(sample_noise(imgs.shape[0])))
			if imgs.shape[0] == TRAIN_BATCH_SIZE:
				real_labels = Variable(Tensor(one_hot_label))
			else:
				real_labels = Variable(Tensor(one_hot_encode(range(int(imgs.shape[0]/2)), range(int(imgs.shape[0]/2), imgs.shape[0]))))
			encoded_imgs = encoder(real_imgs)
			#print(labels)
			#print(np.where(labels == CATEGORIES[0]))
			#print(len(np.where(labels == CATEGORIES[0])))
			#print(np.where(labels == CATEGORIES[0]).shape)
			#print(np.where(labels == CATEGORIES[1]))
			fake_labels = Variable(Tensor(one_hot_encode(np.where(labels == CATEGORIES[0])[0], np.where(labels==CATEGORIES[1])[0])))
			# Measure discriminator's ability to classify real from generated samples
			real_loss = adversarial_loss(discriminator(z, real_labels), valid)

			fake_loss = adversarial_loss(discriminator(encoded_imgs.detach(), fake_labels), fake)
			d_loss = 0.5 * (real_loss + fake_loss)

			d_loss.backward()
			optimizer_D.step()


			if i % N_CRITIC == 0:
				# -----------------
				#  Train Generator
				# -----------------

				optimizer_G.zero_grad()

				encoded_imgs = encoder(real_imgs)
				decoded_imgs = decoder(encoded_imgs)

				# Loss measures generator's ability to fool the discriminator
				g_loss = 0.5 * adversarial_loss(discriminator(encoded_imgs, fake_labels), valid) + 0.5 * pixelwise_loss(
					decoded_imgs, real_imgs
				)

				g_loss.backward()
				optimizer_G.step()

			batches_done = epoch * len(dataloader) + i

			if batches_done % 50 == 0:
				print(
					"[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
					% (epoch, N_EPOCHS, i, len(dataloader), d_loss.item(), g_loss.item())
				)
			
			if batches_done % SAMPLE_INTERVAL == 0:
				name = gen_name(today, batches_done)
				if FIXED_NOISE:
					sample_image(decoder=decoder, n_row=n_row, path=path, name=name, fixed_noise=fixed_noise)
				else:
					sample_image(decoder=decoder, n_row=n_row, path=path, name=name)

			# save losses
			G_losses.append(g_loss.item())
			D_losses.append(d_loss.item())
		#save_model(encoder, epoch, "encoder")
		#save_model(decoder, epoch, "decoder")
		#save_model(discriminator, epoch, "discriminator")
	plot_losses("caae", G_losses, D_losses, CONFIG_AS_STR, today)
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
	print("Saved Encoder to {}".format(save_model(encoder, "caae_encoder", CONFIG_AS_STR, today)))
	print("Saved Decoder to {}".format(save_model(decoder, "caae_decoder", CONFIG_AS_STR, today)))
	print("Saved Discriminator to {}".format(save_model(discriminator, "caae_discriminator", CONFIG_AS_STR, today)))