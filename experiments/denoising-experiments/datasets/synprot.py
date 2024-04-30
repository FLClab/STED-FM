import torch
from torch.utils.data import Dataset
from typing import Tuple, Any
from torchvision import transforms 
from dataclasses import dataclass 
import h5py
import numpy as np

DATAPATH = "/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/DenoisingDataset"

@dataclass 
class SynProtConfiguration:
    num_classes = 1
    criterion: str = "MSELoss"

class ProteinDenoisingDataset(Dataset):
    def __init__(
            self,
            h5file: str,
            transform: Any = None,
            n_channels: int = 1,
            ) -> None:
        self.h5file = h5file 
        self.transform = transform
        self.n_channels = n_channels
        with h5py.File(h5file, "r") as hf:
            self.dataset_size = hf["confocal"][()].shape[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        with h5py.File(self.h5file, "r") as hf:
            confocal = hf["confocal"][idx]
            sted = hf['sted'][idx]
        if self.n_channels == 3:
            confocal = np.tile(confocal[np.newaxis], (3, 1, 1))
            confocal = np.moveaxis(confocal, 0, -1)
            confocal = transforms.ToTensor()(confocal)
            confocal = transforms.Normalize(mean=[0.0695771782959453, 0.0695771782959453, 0.0695771782959453], std=[0.12546228631005282, 0.12546228631005282, 0.12546228631005282])(confocal)
            # sted = np.tile(sted[np.newaxis], (3, 1, 1))
            # sted = np.moveaxis(sted, 0, -1)
            # sted = transforms.ToTensor()(sted)
            # sted = transforms.Normalize(mean=[0.0695771782959453, 0.0695771782959453, 0.0695771782959453], std=[0.12546228631005282, 0.12546228631005282, 0.12546228631005282])(sted)
            sted = transforms.ToTensor()(sted)
        else:
            confocal = transforms.ToTensor()(confocal)
            sted = transforms.ToTensor()(sted)
        return confocal, sted
    

def get_dataset(cfg, **kwargs):
    cfg.dataset_cfg = SynProtConfiguration()
    if kwargs["n_channels"] == 3:
        cfg.in_channels = 3
    cfg.freeze_backbone = True
    train_dataset = ProteinDenoisingDataset(
        h5file=f"{DATAPATH}/train_denoising.hdf5",
        transform=None,
        n_channels=cfg.in_channels
    )
    valid_dataset = ProteinDenoisingDataset(
        h5file=f"{DATAPATH}/valid_denoising.hdf5",
        transform=None,
        n_channels=cfg.in_channels
    )
    test_dataset = ProteinDenoisingDataset(
        h5file=f"{DATAPATH}/test_denoising.hdf5",
        transform=None,
        n_channels=cfg.in_channels
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Valid dataset size: {len(valid_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    return train_dataset, valid_dataset, test_dataset