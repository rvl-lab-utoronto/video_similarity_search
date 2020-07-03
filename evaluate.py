import os
import argparse
import shutil
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
from models.resnet import generate_model
from models.triplet_net import Tripletnet
from datasets import data_loader
import torch.backends.cudnn as cudnn
