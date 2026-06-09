
import numpy
import os
import torch

from torch import nn
from torchvision import transforms

from stedfm.DEFAULTS import BASE_PATH
from stedfm.datasets import FolderDataset
from stedfm.configuration import Configuration
from stedfm.modules.transforms import MinMaxNormalize, CenterCrop, CropToDivisible, Random90DegreeRotation

class SynapticProteinsConfiguration(Configuration):
    num_classes: int = 1
    criterion: str = "MSELoss"

class QuantileNormalize(nn.Module):
    def __init__(self, quantile: float = 0.999, eps: float = 1e-6):
        super().__init__()
        self.quantile = quantile
        self.eps = eps

    def forward(self, x):
        min_val = torch.amin(x, dim=(-2, -1), keepdim=True)

        # Calculate quantile for each channel
        ch_first = x.view(x.shape[0], -1) # Move channels to the end
        q = torch.quantile(ch_first, self.quantile, dim=1, keepdim=True)
        q = q.view(-1, 1, 1) # Reshape back to (C, 1, 1) for broadcasting

        return (x - min_val) / (q - min_val + self.eps)

def get_dataset(name: str, cfg: Configuration, **kwargs):

    rng = numpy.random.default_rng(cfg.get("seed", 42))

    cfg.dataset_cfg = SynapticProteinsConfiguration()

    transform = transforms.Compose([
        MinMaxNormalize(),
        CenterCrop(output_size=224) if "MCMS" not in cfg.backbone_weights else nn.Identity(),
        CropToDivisible(divisor=16),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        Random90DegreeRotation()
    ])
    transform = transforms.Compose([
        QuantileNormalize(quantile=0.999),
        CenterCrop(output_size=224),
        CropToDivisible(divisor=16),
    ])    
    testing_transform = transforms.Compose([
        MinMaxNormalize(),
        CenterCrop(output_size=224) if "MCMS" not in cfg.backbone_weights else nn.Identity(),
        CropToDivisible(divisor=16)
    ])

    if name == "pysoda":
        
        train_dataset = FolderDataset(
            source=os.path.join(BASE_PATH, "evaluation-data", "pysoda"),
            n_channels=cfg.in_channels,
            transform=transform,
            classes=None, # Returns all classes found in the source folder
            **kwargs
        )

    else: 
        raise NotImplementedError(f"`{name}` is not a valid option.")
    return train_dataset

if __name__ == "__main__":

    class Config(Configuration):
        seed: int = 42
        in_channels: int = 1
    cfg = Config()
    dataset = get_dataset(name="pysoda", cfg=cfg)