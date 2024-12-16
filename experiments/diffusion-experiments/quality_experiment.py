import numpy as np 
import matplotlib.pyplot as plt 
from QualityNet.networks import NetTrueFCN 
import argparse 
import torch 
from models.diffusion.diffusion_model import DDPM 
from models.diffusion.denoising.unet import UNet 
from tqdm import trange, tqdm 
import copy 
import random 
import os
from quality_dataset import OptimQualityDataset 
sys.path.insert(0, "../")
from datasets import get_dataset
from DEFAULTS import BASE_PATH 
from model_builder import get_pretrained_model_v2

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="optim")
args = parser.parse_args()

def main():
    pass 

if __name__=="__main__":
    main()

