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
import io
import tarfile

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
    all_classes = []
    with tarfile.open("/home/frbea320/projects/def-flavielc/datasets/FLCDataset/dataset-250k.tar", "r") as handle:
        members = handle.getmembers()
        for m in members:
            buffer = io.BytesIO()
            buffer.write(handle.extractfile(m).read())
            buffer.seek(0)
            data = np.load(buffer, allow_pickle=True)
            metadata = data["metadata"][()]
            protein_id = metadata["protein-id"]
            all_classes.append(protein_id)
    uniques, counts = np.unique(all_classes, return_counts=True)
    for u, c in zip(uniques, counts):
        print(f"{u}: {c}")
            


    


if __name__=="__main__":
    main()