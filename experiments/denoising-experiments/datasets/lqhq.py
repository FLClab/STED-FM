
import os

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

def get_dataset(name: str, cfg: Configuration, **kwargs) -> Dataset:
    cfg.dataset_cfg = LQHQConfiguration()

    if name == "lqhq":
        path = os.path.join(BASE_PATH, "denoising-data", "lqhq")
        training_dataset = LQHQDenoisingDataset(
            tarpath=os.path.join(path, "train-dataset.tar"), 
            n_channels=cfg.in_channels, **kwargs)
        validation_dataset = LQHQDenoisingDataset(
            tarpath=os.path.join(path, "valid-dataset.tar"), 
            n_channels=cfg.in_channels, **kwargs)
        testing_dataset = LQHQDenoisingDataset(
            tarpath=os.path.join(path, "test-dataset.tar"), 
            n_channels=cfg.in_channels, **kwargs)
        return SplitViewsDataset(training_dataset), \
               SplitViewsDataset(validation_dataset), \
               SplitViewsDataset(testing_dataset)

    raise NotImplementedError(f"Dataset '{name}' is not implemented.")
