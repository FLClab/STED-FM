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
    model, cfg = get_pretrained_model_v2(
        name="mae-lightning-small",
        weights="MAE_SMALL_STED",
        path=None,
        mask_ratio=0.0, 
        pretrained=False,
        in_channels=1,
        as_classifier=False,
    )
    print(model)

        
        


if __name__=="__main__":
    main()
