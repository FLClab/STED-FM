import torch 
from torch.utils.data import Dataset
from typing import Tuple
from torchvision import transforms
from dataclasses import dataclass
import h5py
import numpy as np

DATAPATH = "/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/TheresaProteins"

@dataclass
class SynProtConfiguration:
    num_classes=1
    criterion: str = "MSELoss"

class ProteinSegmentationDataset(Dataset):
    def __init__(
        self,
        h5file: str,
        transform=None,
        n_channels=1,
    ) -> None:
        self.h5file = h5file
        self.transform = transform 
        self.n_channels = n_channels 
        with h5py.File(h5file, "r") as hf:
            self.dataset_size = hf["images"][()].shape[0] 

    def __len__(self):
        return self.dataset_size 


    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.Tensor]:
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
        mask = transforms.ToTensor()(mask)
        return img, {"label": mask}

def get_dataset(cfg, **kwargs):
    cfg.dataset_cfg = SynProtConfiguration()
    if kwargs['n_channels'] == 3:
        cfg.in_channels = 3
    cfg.freeze_backbone = True
    train_dataset = ProteinSegmentationDataset(
        h5file=f"{DATAPATH}/train_segmentation.hdf5",
        transform=None, 
        n_channels=cfg.in_channels
    )
    valid_dataset = ProteinSegmentationDataset(
        h5file=f"{DATAPATH}/valid_segmentation.hdf5",
        transform=None, 
        n_channels=cfg.in_channels
    )
    test_dataset = ProteinSegmentationDataset(
        h5file=f"{DATAPATH}/test_segmentation.hdf5",
        transform=None, 
        n_channels=cfg.in_channels
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Valid dataset size: {len(valid_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    return train_dataset, valid_dataset, test_dataset
