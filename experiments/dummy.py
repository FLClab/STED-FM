import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import random 
import json 
from tqdm import tqdm, trange 
import argparse 
from loaders import get_dataset
import os
from model_builder import get_pretrained_model_v2

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="peroxisome")
parser.add_argument("--num", type=int, default=20)
args = parser.parse_args()


def main():
#     train_loader, _, _ = get_dataset(
#         name="peroxisome",
#         training=True
#     )
#     print(f"Peroxisome: {len(train_loader.dataset)}")

#     train_loader, _, _ = get_dataset(
#         name="polymer-rings",
#         training=True
#     )
#     print(f"Polymer Rings: {len(train_loader.dataset)}")

    train_loader, _, _ = get_dataset(
        name="dl-sim",
        training=True
    )
    print(f"DL Sim: {len(train_loader.dataset)}")
    

        
        


if __name__=="__main__":
    main()
