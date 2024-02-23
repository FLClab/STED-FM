import numpy as np
from torch.utils.data import DataLoader
from typing import List
from datasets.datasets import ProteinDataset

PROTEINS_PATH = "/home/frbea320/scratch/Datasets/FLCDataset/TheresaProteins"
H5SIZE = 67436

def load_theresa_proteins(path: str = PROTEINS_PATH, class_ids: List = None, class_type: str = "protein", batch_size: int = 256):
    train_dataset = ProteinDataset(
        h5file=f"{path}/theresa_proteins.hdf5",
        class_ids=None,
        class_type=class_type
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=6,
    )
    return train_loader
