import torch 
from torch.utils.data import DataLoader 
from tqdm import tqdm 
import numpy as np 
import sys
import argparse  
sys.path.insert(0, "../")
from DEFAULTS import BASE_PATH 
from loaders import get_dataset 
from model_builder import get_model 

parser = argparse.ArgumentParser() 
parser.add_argument("--dataset", type=str, default="optim")
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument
