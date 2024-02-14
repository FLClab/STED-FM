import numpy as np
from torch.utils.data import Dataset, DataLoader
from data.datasets import TarFLCDataset


TARPATH = "/home/frbea320/scratch/Datasets/FLCDataset/dataset.tar"

def tar_dataloader(path: str = TARPATH):
    dataset = TarFLCDataset(tar_path=path)
    dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True, drop_last=False, num_workers=6)
    return dataloader