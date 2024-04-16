import numpy as np
from torch.utils.data import DataLoader
from typing import List
from datasets.datasets import ProteinDataset

PROTEINS_PATH = "/home/frbea320/scratch/Datasets/FLCDataset/TheresaProteins"
H5SIZE = 67436

def fewshot_loader(
        path: str = PROTEINS_PATH, 
        class_ids: List = None,
        batch_size: int = 256,
        validation_size: float = 0.10,
        class_type: str = "protein",
        n_channels: int = 1
        ) -> DataLoader:
        indices = np.arange(0, H5SIZE, 1)
        np.random.shuffle(indices)
        split = int( (H5SIZE) * (1 - validation_size))
        train_indices = indices[:split]
        valid_indices = indices[split:]
        train_dataset = ProteinDataset(
            h5file=f"{path}/theresa_proteins.hdf5",
            class_ids=None,
            class_type=class_type,
            n_channels=n_channels,
            indices=train_indices
        )
        valid_dataset = ProteinDataset(
            h5file=f"{path}/theresa_proteins.hdf5",
            class_ids=None,
            class_type=class_type,
            n_channels=n_channels,
            indices=valid_indices
        )
        print(f"Training size: {len(train_dataset)}")
        print(f"Validation size: {len(valid_dataset)}")
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=6,
        )
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=6,
        )

        return train_loader, valid_loader

def load_theresa_proteins(
        path: str = PROTEINS_PATH, 
        class_ids: List = None, 
        class_type: str = "protein", 
        batch_size: int = 256, 
        n_channels: int = 1) -> DataLoader:
    train_dataset = ProteinDataset(
        h5file=f"{path}/theresa_proteins.hdf5",
        class_ids=None,
        class_type=class_type,
        n_channels=n_channels
    )
    print(f"Dataset size: {len(train_dataset)}")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=6,
    )
    return train_loader
