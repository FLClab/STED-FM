
import os
import random
import numpy

from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from stedfm.DEFAULTS import BASE_PATH
from stedfm.configuration import Configuration
from stedfm.datasets import LQHQDenoisingDataset

class LQHQConfiguration(Configuration):
    num_classes: int = 1
    criterion: str = "MSELoss"

class SplitViewsDataset(Dataset):
    """
    A dataset that simply splits the channels of a given dataset into two views.
    It assumes that the input dataset has exactly two channels.
    """
    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset
        assert self.base_dataset[0][0].shape[0] == 2, "The base dataset must have exactly two channels."

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, metadata = self.base_dataset[idx]
        view1 = img[0:1, :, :]  # First channel as first view
        view2 = img[1:2, :, :]  # Second channel as second view
        return view1, view2
    
class Random90DegreeRotation(nn.Module):
    def __init__(self):
        super().__init__()
        self.degrees = [0, 90, 180, 270]

    def forward(self, img):
        angle = random.choice(self.degrees)
        return transforms.functional.rotate(img, angle)

def get_dataset(name: str, cfg: Configuration, **kwargs) -> Dataset:
    cfg.dataset_cfg = LQHQConfiguration()

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        Random90DegreeRotation()
    ])

    if name == "jmb-lqhq":
        path = os.path.join(BASE_PATH, "denoising-data", "jmb-lqhq")
        training_dataset = LQHQDenoisingDataset(
            tarpath=os.path.join(path, "train-dataset.tar"),
            n_channels=cfg.in_channels, 
            transform=transform,
            **kwargs)
        validation_dataset = LQHQDenoisingDataset(
            tarpath=os.path.join(path, "valid-dataset.tar"), 
            n_channels=cfg.in_channels, **kwargs)
        testing_dataset = LQHQDenoisingDataset(
            tarpath=os.path.join(path, "test-dataset.tar"), 
            n_channels=cfg.in_channels, **kwargs)
    elif name == "kt-lqhq-vgat":
        path = os.path.join(BASE_PATH, "denoising-data", "kt-lqhq")
        training_dataset = LQHQDenoisingDataset(
            tarpath=os.path.join(path, "train-dataset.tar"),
            n_channels=cfg.in_channels, 
            transform=transform,
            classes=["VGAT_ATTO490LS"],
            **kwargs)
        validation_dataset = LQHQDenoisingDataset(
            tarpath=os.path.join(path, "valid-dataset.tar"), 
            classes=["VGAT_ATTO490LS"],
            n_channels=cfg.in_channels, **kwargs)
        testing_dataset = LQHQDenoisingDataset(
            tarpath=os.path.join(path, "test-dataset.tar"), 
            classes=["VGAT_ATTO490LS"],
            n_channels=cfg.in_channels, **kwargs)
    elif name == "kt-lqhq-gephyrin":
        path = os.path.join(BASE_PATH, "denoising-data", "kt-lqhq")
        training_dataset = LQHQDenoisingDataset(
            tarpath=os.path.join(path, "train-dataset.tar"),
            n_channels=cfg.in_channels, 
            classes=["Gephyrin_STARRED"],
            transform=transform,
            **kwargs)
        validation_dataset = LQHQDenoisingDataset(
            tarpath=os.path.join(path, "valid-dataset.tar"), 
            classes=["Gephyrin_STARRED"],
            n_channels=cfg.in_channels, **kwargs)
        testing_dataset = LQHQDenoisingDataset(
            tarpath=os.path.join(path, "test-dataset.tar"), 
            classes=["Gephyrin_STARRED"],
            n_channels=cfg.in_channels, **kwargs)
    elif name == "kt-lqhq":
        path = os.path.join(BASE_PATH, "denoising-data", "kt-lqhq")
        training_dataset = LQHQDenoisingDataset(
            tarpath=os.path.join(path, "train-dataset.tar"),
            n_channels=cfg.in_channels, 
            classes=["VGAT_ATTO490LS", "Gephyrin_STARRED"],
            transform=transform,
            **kwargs)
        validation_dataset = LQHQDenoisingDataset(
            tarpath=os.path.join(path, "valid-dataset.tar"), 
            classes=["VGAT_ATTO490LS", "Gephyrin_STARRED"],
            n_channels=cfg.in_channels, **kwargs)
        testing_dataset = LQHQDenoisingDataset(
            tarpath=os.path.join(path, "test-dataset.tar"), 
            classes=["VGAT_ATTO490LS", "Gephyrin_STARRED"],
            n_channels=cfg.in_channels, **kwargs)
    else:
        raise NotImplementedError(f"Dataset '{name}' is not implemented.")
        
    # # Export images as tiff for visualization
    # import tifffile
    # os.makedirs(os.path.join(path, "exported", "train", "raw_256"), exist_ok=True)
    # os.makedirs(os.path.join(path, "exported", "train", "gt_256"), exist_ok=True)

    # os.makedirs(os.path.join(path, "exported", "test", "raw_256"), exist_ok=True)
    # os.makedirs(os.path.join(path, "exported", "test", "gt_256"), exist_ok=True)

    # stack = []
    # for i in range(len(training_dataset)):
    #     images, _ = training_dataset[i]
    #     images = transforms.functional.resize(images, (256, 256), interpolation=transforms.InterpolationMode.NEAREST)
    #     stack.append(images.numpy())
    # for i in range(len(validation_dataset)):
    #     images, _ = validation_dataset[i]
    #     images = transforms.functional.resize(images, (256, 256), interpolation=transforms.InterpolationMode.NEAREST)
    #     stack.append(images.numpy())
    # stack = numpy.stack(stack, axis=0)
    # for i in range(stack.shape[0]):
    #     tifffile.imwrite(os.path.join(path, "exported", "train", "raw_256", f"img_{i:04d}.tif"), stack[i, 0, :, :])
    #     tifffile.imwrite(os.path.join(path, "exported", "train", "gt_256", f"img_{i:04d}.tif"), stack[i, 1, :, :])
    # # tifffile.imwrite(os.path.join(path, "exported", "train", "raw", "stack.tif"), stack[:, 0, :, :])
    # # tifffile.imwrite(os.path.join(path, "exported", "train", "gt", "stack.tif"), stack[:, 1, :, :])

    # stack = []
    # for i in range(len(testing_dataset)):
    #     images, _ = testing_dataset[i]
    #     images = transforms.functional.resize(images, (256, 256), interpolation=transforms.InterpolationMode.NEAREST)
    #     stack.append(images.numpy())
    # stack = numpy.stack(stack, axis=0)
    # for i in range(stack.shape[0]):
    #     tifffile.imwrite(os.path.join(path, "exported", "test", "raw_256", f"img_{i:04d}.tif"), stack[i, 0, :, :])
    #     tifffile.imwrite(os.path.join(path, "exported", "test", "gt_256", f"img_{i:04d}.tif"), stack[i, 1, :, :])

    # exit()

    return SplitViewsDataset(training_dataset), \
            SplitViewsDataset(validation_dataset), \
            SplitViewsDataset(testing_dataset)

    raise NotImplementedError(f"Dataset '{name}' is not implemented.")
