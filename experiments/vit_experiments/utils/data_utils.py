import numpy as np
from torch.utils.data import Dataset, DataLoader
from data.datasets import TarFLCDataset, CTCDataset
from typing import List, Any


TARPATH = "/home/frbea320/scratch/Datasets/FLCDataset/dataset.tar"

def tar_dataloader(transform, path: str = TARPATH):
    dataset = TarFLCDataset(tar_path=path, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=6)
    return dataloader

def ctc_loader(transform, path: str):
    dataset = CTCDataset(h5file=path, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=6)
    return dataloader