
import os
import random
import numpy
import torch

from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from stedfm.DEFAULTS import BASE_PATH
from stedfm.configuration import Configuration
from stedfm.datasets import FolderDataset

class reg3bConfiguration(Configuration):
    num_classes: int = 1
    criterion: str = "MSELoss"

class Random90DegreeRotation(nn.Module):
    def __init__(self):
        super().__init__()
        self.degrees = [0, 90, 180, 270]

    def forward(self, img):
        angle = random.choice(self.degrees)
        return transforms.functional.rotate(img, angle)    

class CropToDivisible(nn.Module):
    def __init__(self, divisor: int = 16):
        super().__init__()
        self.divisor = divisor

    def forward(self, img):
        H, W = img.shape[-2], img.shape[-1]
        new_H = (H // self.divisor) * self.divisor
        new_W = (W // self.divisor) * self.divisor
        if img.ndim == 3:
            img = img[:, :new_H, :new_W]
        else:
            img = img[:new_H, :new_W]
        return img

class CenterCrop(nn.Module):
    def __init__(self, output_size: int = 224):
        super().__init__()
        self.output_size = output_size

    def forward(self, img):
        H, W = img.shape[-2], img.shape[-1]
        top = (H - self.output_size) // 2
        left = (W - self.output_size) // 2
        if img.ndim == 3:
            img = img[:, top:top+self.output_size, left:left+self.output_size]
        else:
            img = img[top:top+self.output_size, left:left+self.output_size]
        return img

class MinMaxNormalize(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, img):
        min_val = torch.amin(img, dim=(-2, -1), keepdim=True)
        max_val = torch.amax(img, dim=(-2, -1), keepdim=True)
        normalized = (img - min_val) / (max_val - min_val + self.eps)
        return normalized

def get_dataset(name: str, cfg: Configuration, **kwargs) -> Dataset:

    rng = numpy.random.default_rng(cfg.get('seed', 42))

    cfg.dataset_cfg = reg3bConfiguration()

    transform = transforms.Compose([
        MinMaxNormalize(),
        CenterCrop(output_size=224) if "MCMS" not in cfg.backbone_weights else nn.Identity(),
        CropToDivisible(divisor=16),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        Random90DegreeRotation()
    ])
    testing_transform = transforms.Compose([
        MinMaxNormalize(),
        CenterCrop(output_size=224) if "MCMS" not in cfg.backbone_weights else nn.Identity(),
        CropToDivisible(divisor=16)
    ])

    if name == "reg3b":
        print("Loading reg3b dataset...")
        path = os.path.join(BASE_PATH, "evaluation-data", "reg3b", "processed")
        training_dataset = FolderDataset(
            source=os.path.join(path, "train"),
            n_channels=cfg.in_channels,
            transform=transform,
            classes=None, # Returns all classes found in the source folder
            **kwargs)

        num_samples = len(training_dataset)
        random_indices = rng.choice(num_samples, size=int(0.1 * num_samples), replace=False)
        validation_dataset = torch.utils.data.Subset(training_dataset, random_indices)
        training_dataset = torch.utils.data.Subset(training_dataset, list(set(range(num_samples)) - set(random_indices)))

        testing_dataset = FolderDataset(
            source=os.path.join(path, "test"),
            n_channels=cfg.in_channels,
            transform=testing_transform, # We apply the same cropping to the test set, but no other augmentations
            classes=None, # Returns all classes found in the source folder
            **kwargs)
    else:
        raise NotImplementedError(f"`{name}` is not a valid option.")
    
    return training_dataset, validation_dataset, testing_dataset

