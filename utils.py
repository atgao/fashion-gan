import os
import time
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

def save_model(model, model_type, config, date):
	os.makedirs("model", exist_ok=True)
	save_path = "model/{}.pkl".format(gen_name(model_type, config, date))
	torch.save(model.state_dict(), save_path)
	return save_path

def load_model(model_type, config, date, device):
	model_path = "model/{}.pkl".format(gen_name(model_type, config, date))
	return torch.load(model_path, map_location=device)

def plot_losses(model_name, G_losses, D_losses, config, date, g_step = 1, d_step = 1, show=True, save=True):
	plt.figure(figsize=(10,5))
	plt.title("Generator and Discriminator Loss During Training")
	G_x = range(0, len(G_losses), g_step)
	D_x = range(0, len(D_losses), d_step)
	plt.plot(G_x, G_losses,label="G")
	plt.plot(D_x, D_losses,label="D")
	plt.xlabel("iterations")
	plt.ylabel("Loss")
	plt.legend()
	if save:
		os.makedirs("plots", exist_ok=True)
		plt.savefig('plots/%s.png' % gen_name(model_name, config, date))

	if show:
		plt.show(block=False)

def gen_name(model_name, *args):
	name = model_name
	for i in range(len(args)):
		name = name + "_" + str(args[i])
	return name
