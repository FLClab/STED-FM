"""
Script mostly used for displaying quick info or downloading torchvision models on login nodes (w/ internet)
"""
import torch
import numpy as np
from model_builder import get_pretrained_model, get_pretrained_model_v2, get_base_model
from timm.models.vision_transformer import vit_small_patch16_224, vit_base_patch16_224, vit_tiny_patch16_224
import argparse 
import torchvision
from torchinfo import summary
from loaders import get_dataset
from tqdm import tqdm
import h5py

PATH = "/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/TheresaProteins/"

def list_of_numbers(arg):
    return [int(item) for item in arg.split(',')]

def reset_labels(idx):
    if idx > 1:
        return idx - 1
    else:
        return idx


def main():
    # model, cfg = get_base_model(
    #     name="convnext-small",
    # )
    model = vit_tiny_patch16_224(in_chans=1)
    
    summary(model)

if __name__=="__main__":
    main()