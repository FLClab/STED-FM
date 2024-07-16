import torch 
from torch.utils.data import Dataset
from typing import Tuple, Callable
from torchvision import transforms
from dataclasses import dataclass
import h5py
import numpy as np

DATAPATH = "/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/zooniverse"

@dataclass 
class ZooniverseConfiguration:
    num_classes: int = 6
    criterion: str = "MSELoss"
    in_channels: int = 1


class SemanticZooniverseDataset(Dataset):
    def __init__(
            self,
            h5file: str,
            transform: Callable = None,
            n_channels: int = 1
    ) -> None:
        self.h5file = h5file 
        self.transform = transform
        self.n_channels = n_channels
        with h5py.File(h5file, "r") as handle:
            self.dataset_size = handle["images"][()].shape[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        with h5py.File(self.h5file, "r") as hf:
            img = hf["images"][idx]
            mask = hf["masks"][idx]
        if self.n_channels == 3:
            img = np.tile(img[np.newaxis], (3, 1, 1))
            img = np.moveaxis(img, 0, -1)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize(mean=[0.0695771782959453, 0.0695771782959453, 0.0695771782959453], std=[0.12546228631005282, 0.12546228631005282, 0.12546228631005282])(img)
        else:
            img = transforms.ToTensor()(img)
        mask = torch.tensor(mask, dtype=torch.float32)
        return img, mask

def get_dataset(cfg, **kwargs):
    cfg.dataset_cfg = ZooniverseConfiguration()
    if kwargs["n_channels"] == 3:
        cfg.in_channels=3
    cfg.freeze_backbone = True 
    train_dataset = SemanticZooniverseDataset(
        h5file=f"{DATAPATH}/train.hdf5",
        transform=None,
        n_channels=cfg.in_channels
    )
    valid_dataset = SemanticZooniverseDataset(
        h5file=f"{DATAPATH}/valid.hdf5",
        transform=None,
        n_channels=cfg.in_channels,
    )
    test_dataset = SemanticZooniverseDataset(
        h5file=f"{DATAPATH}/test.hdf5",
        transform=None,
        n_channels=cfg.in_channels,
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Valid dataset size: {len(valid_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    return train_dataset, valid_dataset, test_dataset