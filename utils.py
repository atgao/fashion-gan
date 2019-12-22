import os
import time
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F


def save_model(model, epoch, model_type):
	os.makedirs("model", exist_ok=True)
	save_path = "model/{}_{}.pkl".format(model_type,epoch)
	torch.save(model.state_dict(), save_path)
	return save_path

def load_model():
	pass 
