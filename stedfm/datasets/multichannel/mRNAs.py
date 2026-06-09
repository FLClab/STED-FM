
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
from stedfm.modules.transforms import MinMaxNormalize, CenterCrop, CropToDivisible, Random90DegreeRotation

class mRNAsConfiguration(Configuration):
    num_classes: int = 1
    criterion: str = "MSELoss"

def get_dataset(name: str, cfg: Configuration, **kwargs) -> Dataset:

    rng = numpy.random.default_rng(cfg.get('seed', 42))

    cfg.dataset_cfg = mRNAsConfiguration()

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

    if name == "mRNAs-3b":
        print("Loading mRNAs-3b dataset...")
        path = os.path.join(BASE_PATH, "evaluation-data", "mRNAs", "processed", "figure3B")
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
        
    if name == "mRNAs-3b-KD":
        print("Loading mRNAs-3b dataset...")
        path = os.path.join(BASE_PATH, "evaluation-data", "mRNAs", "processed", "figure3B")
        training_dataset = FolderDataset(
            source=os.path.join(path, "train"),
            n_channels=cfg.in_channels,
            transform=transform,
            classes=['CYB KD', 'ATP6 KD', 'CO1 KD', 'ND1 KD'], # Returns all classes found in the source folder
            **kwargs)

        num_samples = len(training_dataset)
        random_indices = rng.choice(num_samples, size=int(0.1 * num_samples), replace=False)
        validation_dataset = torch.utils.data.Subset(training_dataset, random_indices)
        training_dataset = torch.utils.data.Subset(training_dataset, list(set(range(num_samples)) - set(random_indices)))

        testing_dataset = FolderDataset(
            source=os.path.join(path, "test"),
            n_channels=cfg.in_channels,
            transform=testing_transform, # We apply the same cropping to the test set, but no other augmentations
            classes=['CYB KD', 'ATP6 KD', 'CO1 KD', 'ND1 KD'], # Returns all classes found in the source folder
            **kwargs)
                
    elif name == "mRNAs-3d":
        print("Loading mRNAs-3d dataset...")
        path = os.path.join(BASE_PATH, "evaluation-data", "mRNAs", "processed", "figure3D")
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
            transform=testing_transform,
            classes=None, # Returns all classes found in the source folder
            **kwargs)

    elif name == "mRNAs-3f":
        print("Loading mRNAs-3f dataset...")
        path = os.path.join(BASE_PATH, "evaluation-data", "mRNAs", "processed", "figure3F")
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
            transform=testing_transform,
            classes=None, # Returns all classes found in the source folder
            **kwargs)
    else:
        raise NotImplementedError(f"`{name}` is not a valid option.")
    
    return training_dataset, validation_dataset, testing_dataset

