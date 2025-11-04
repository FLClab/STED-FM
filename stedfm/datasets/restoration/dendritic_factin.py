import os 
import numpy as np 
import torch
import random
from torch import nn 
from torch.utils.data import Dataset 
from torchvision import transforms 
from typing import List, Optional, Callable, Tuple
from stedfm.DEFAULTS import BASE_PATH
import glob
from stedfm.configuration import Configuration
from stedfm.datasets import RestorationFolderDataset

class DendriticFActinConfiguration(Configuration):
    num_classes: int = 1
    criterion: str = "MSELoss"

class Random90DegreeRotation(nn.Module):
    def __init__(self):
        super().__init__()
        self.degrees = [0, 90, 180, 270]

    def forward(self, img):
        angle = random.choice(self.degrees)
        return transforms.functional.rotate(img, angle)

class DendriticFActinDataset(Dataset):
    def __init__(
        self, 
        path: str,
        n_channels: int = 1,
        crop_size: int = 224, 
        transform: Optional[Callable] = None,
        **kwargs
    ) -> None:
        self.n_channels = n_channels
        self.crop_size = crop_size
        self.transform = transform
        self.files = glob.glob(f"{path}/*.tif")

    
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = tifffile.imread(self.files[idx])
        confocal, sted = data[0, :, :], data[1, :, :] 
        confocal = confocal / 255.0 
        sted = sted / 255.0  
        confocal = torch.tensor(confocal[np.newaxis, ...], dtype=torch.float32)
        sted = torch.tensor(sted[np.newaxis, ...], dtype=torch.float32)
        cat = torch.cat([confocal, sted], dim=0)
        cat = self.transform(cat) if self.transform is not None else cat
        confocal, sted = cat[0:1], cat[1:2]
        return confocal, sted

def get_dataset(name: str, cfg: Configuration, **kwargs) -> Dataset:
    rng = np.random.default_rng(cfg.get('seed', 42)) 

    cfg.dataset_cfg = DendriticFActinConfiguration()

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        Random90DegreeRotation()
    ])

    test_transform = transforms.Compose([
        transforms.RandomCrop(224),
    ])

    split = kwargs.get("split", "train")
    train_path = os.path.join(BASE_PATH, "Datasets", "train")
    valid_path = os.path.join(BASE_PATH, "Datasets", "valid")
    test_path = os.path.join(BASE_PATH, "Datasets", "test")
    train_dataset = DendriticFActinDataset(train_path, transform=train_transform, **kwargs)
    valid_dataset = DendriticFActinDataset(valid_path, transform=train_transform, **kwargs)
    test_dataset = DendriticFActinDataset(test_path, transform=test_transform, **kwargs)
    return train_dataset, valid_dataset, test_dataset
