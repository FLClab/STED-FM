
import os
import random
import numpy
import torch

from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from stedfm.DEFAULTS import BASE_PATH
from stedfm.configuration import Configuration
from stedfm.datasets import RestorationFolderDataset

class SRConfiguration(Configuration):
    num_classes: int = 1
    criterion: str = "MSELoss"

class Random90DegreeRotation(nn.Module):
    def __init__(self):
        super().__init__()
        self.degrees = [0, 90, 180, 270]

    def forward(self, img):
        angle = random.choice(self.degrees)
        return transforms.functional.rotate(img, angle)

def get_dataset(name: str, cfg: Configuration, **kwargs) -> Dataset:

    rng = numpy.random.default_rng(cfg.get('seed', 42))

    cfg.dataset_cfg = SRConfiguration()

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        Random90DegreeRotation()
    ])

    if name == "3d-rcan-mt":
        print("Loading 3D-RCAN-MT dataset...")
        path = os.path.join(BASE_PATH, "super-resolution-data", "3d-rcan", "Confocal_2_STED", "Microtubule")
        training_dataset = RestorationFolderDataset(
            source=os.path.join(path, "Training", "raw"),
            target=os.path.join(path, "Training", "gt"),
            n_channels=cfg.in_channels,
            transform=transform,
            **kwargs)
        
        num_samples = len(training_dataset)
        random_indices = rng.choice(num_samples, size=int(0.1 * num_samples), replace=False)
        validation_dataset = torch.utils.data.Subset(training_dataset, random_indices)
        training_dataset = torch.utils.data.Subset(training_dataset, list(set(range(num_samples)) - set(random_indices)))
        
        testing_dataset = RestorationFolderDataset(
            source=os.path.join(path, "test"),
            target=os.path.join(path, "test", "testgt"),
            n_channels=cfg.in_channels,
            transform=transform,
            **kwargs)
    elif name == "3d-rcan-npc":
        print("Loading 3D-RCAN-NPC dataset...")
        path = os.path.join(BASE_PATH, "super-resolution-data", "3d-rcan", "Confocal_2_STED", "Nuclear_Pore_complex")
        training_dataset = RestorationFolderDataset(
            source=os.path.join(path, "Training", "raw"),
            target=os.path.join(path, "Training", "gt"),
            n_channels=cfg.in_channels,
            transform=transform,
            **kwargs)
        
        num_samples = len(training_dataset)
        random_indices = rng.choice(num_samples, size=int(0.1 * num_samples), replace=False)
        validation_dataset = torch.utils.data.Subset(training_dataset, random_indices)
        training_dataset = torch.utils.data.Subset(training_dataset, list(set(range(num_samples)) - set(random_indices)))

        testing_dataset = RestorationFolderDataset(
            source=os.path.join(path, "test"),
            target=os.path.join(path, "test", "testgt"),
            n_channels=cfg.in_channels,
            transform=transform,
            **kwargs)
    else:
        raise NotImplementedError(f"`{name}` is not a valid option.")
    
    return training_dataset, validation_dataset, testing_dataset