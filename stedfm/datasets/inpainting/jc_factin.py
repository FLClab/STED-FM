
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

class GenerationConfiguration(Configuration):
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

    cfg.dataset_cfg = GenerationConfiguration()

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        Random90DegreeRotation()
    ])

    if name == "jc-factin-tau":
        print("Loading JC-FACTIN-TAU dataset...")
        path = "/home-local/Actin/jchabbert/inpainting-experiments/actin-tau-dataset"
        training_dataset = RestorationFolderDataset(
            source=os.path.join(path, "split_data", "train", "source"),
            target=os.path.join(path, "split_data", "train", "target"),
            n_channels=cfg.in_channels, 
            transform=transform,
            use_foreground=True,
            **kwargs)
        
        validation_dataset = RestorationFolderDataset(
            source=os.path.join(path, "split_data", "val", "source"),
            target=os.path.join(path, "split_data", "val", "target"),
            n_channels=cfg.in_channels, 
            transform=transform,
            use_foreground=True,
            **kwargs)

        testing_dataset = RestorationFolderDataset(
            source=os.path.join(path, "split_data", "test", "source"),
            target=os.path.join(path, "split_data", "test", "target"),
            n_channels=cfg.in_channels, 
            transform=None,
            use_foreground=True,
            **kwargs)
    else:
        raise NotImplementedError(f"`{name}` is not a valid option.")

    # for name, dataset in zip(("train", "valid", "test"), [training_dataset, validation_dataset, testing_dataset]):
    #     X_samples, y_samples = [], []
    #     for X, y in dataset:
    #         X = X.numpy().squeeze()
    #         y = y.numpy().squeeze()
    #         X_samples.append(X)
    #         y_samples.append(y)
    #     X_samples = numpy.stack(X_samples, axis=0)
    #     y_samples = numpy.stack(y_samples, axis=0)

    #     import tifffile
    #     tifffile.imwrite(os.path.join(path, f"{name}_raw.tif"), X_samples)
    #     tifffile.imwrite(os.path.join(path, f"{name}_gt.tif"), y_samples)
    # exit()

    return training_dataset, validation_dataset, testing_dataset