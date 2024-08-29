"""
Script mostly used for displaying quick info or downloading torchvision models on login nodes (w/ internet)
"""
import torch
import numpy as np
from model_builder import get_pretrained_model_v2, get_base_model
from timm.models.vision_transformer import vit_small_patch16_224, vit_base_patch16_224, vit_tiny_patch16_224, vit_large_patch16_224

from torchinfo import summary
from loaders import get_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List
from torch.utils.data import DataLoader
from timm.models.vision_transformer import default_cfgs
from datasets import get_dataset

PATH = "/home/frbea320/scratch/Datasets/SynapticProteins/dataset"
OUTPATH = "/home/frbea320/scratch/Datasets/FLCDataset/TheresaProteins"
THRESHOLD = 0.001
protein_dict = {
    "Bassoon": 0,
    "Homer": 1,
    "NKCC2": 2,
    "Rim": 3,
    "PSD95": 4,
}

condition_dict = {
    "0MgGlyBic": 0,
    "Block": 1,
    "GluGly": 2,
    "KCl": 3,
    "4hTTX": 4,
    "24hTTX": 5,
    "48hTTX": 6,
    "naive": 7,
}

def main():
    device = torch.device("cpu")
    dataset = get_dataset(name="JUMP", path="/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/JUMP_CP/jump.tar")
    # dataloader = DataLoader(dataset=dataset, batch_size=256, drop_last=False, num_workers=1, shuffle=True)

    for i in range(20):
        img = dataset[i]
        print(type(img))
        print(img.shape)
        i += 1
        img = img.to(device)
        img = torch.squeeze(0).cpu().detach().numpy()
        fig = plt.figure()
        plt.imshow(img, cmap='hot')
        fig.savefig(f"./dummy/temp{i}.png", dpi=1200, bbox_inches='tight')
        plt.close(fig)
        if i > 20:
            exit()

    


if __name__=="__main__":
    main()