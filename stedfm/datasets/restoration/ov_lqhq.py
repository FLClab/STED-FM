
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

class LQHQConfiguration(Configuration):
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

    cfg.dataset_cfg = LQHQConfiguration()

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        Random90DegreeRotation()
    ])

    if name == "ov-lqhq-live-mito":
        print("Loading OV-LQHQ-LIVE-MITO dataset...")
        mu, std = [0.049] * 3, [0.086] * 3
        path = os.path.join(BASE_PATH, "denoising-data", "ov-lqhq-live-mito", "live_cell_mitochondria_u2os_tom20_halotag7_dm_sir")
        training_dataset = RestorationFolderDataset(
            source=os.path.join(path, "test_and_training_data_2", "low_intensity_images"),
            target=os.path.join(path, "test_and_training_data_2", "ground_truth_images"),
            n_channels=cfg.in_channels, 
            transform=transform,
            mu=mu,
            std=std,
            **kwargs)
        
        num_samples = len(training_dataset)
        random_indices = rng.choice(num_samples, size=int(0.1 * num_samples), replace=False)
        validation_dataset = torch.utils.data.Subset(training_dataset, random_indices)
        training_dataset = torch.utils.data.Subset(training_dataset, list(set(range(num_samples)) - set(random_indices)))

        testing_dataset = RestorationFolderDataset(
            source=os.path.join(path, "test_and_training_data_1", "low_intensity_images"),
            target=os.path.join(path, "test_and_training_data_1", "ground_truth_images"),
            n_channels=cfg.in_channels, 
            transform=None,
            mu=mu,
            std=std,
            **kwargs)
    elif name == "ov-lqhq-mt":
        print("Loading OV-LQHQ-MT dataset...")
        mu, std = [0.058] * 3, [0.091] * 3
        path = os.path.join(BASE_PATH, "denoising-data", "ov-lqhq-mt", "fixed_cell_microtubule_u2os_alphatubulin_star635p")
        training_dataset = RestorationFolderDataset(
            source=os.path.join(path, "training_data", "low_intensity_image_patches"),
            target=os.path.join(path, "training_data", "ground_truth_image_patches"),
            n_channels=cfg.in_channels, 
            transform=transform,
            mu=mu,
            std=std,
            **kwargs)
        
        num_samples = len(training_dataset)
        random_indices = rng.choice(num_samples, size=int(0.1 * num_samples), replace=False)
        validation_dataset = torch.utils.data.Subset(training_dataset, random_indices)
        training_dataset = torch.utils.data.Subset(training_dataset, list(set(range(num_samples)) - set(random_indices)))

        testing_dataset = RestorationFolderDataset(
            source=os.path.join(path, "test_data", "low_intensity_image_patches"),
            target=os.path.join(path, "test_data", "ground_truth_image_patches"),
            n_channels=cfg.in_channels, 
            transform=None,
            mu=mu,
            std=std,
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

    return training_dataset, validation_dataset, testing_dataset