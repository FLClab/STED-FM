
import torch
import torchvision
from dataclasses import dataclass

@dataclass
class ResNetConfiguration:
    
    backbone: str = "resnet"
    batch_size: int = 256
    dim: int = 512

def get_backbone(name: str) -> torch.nn.Module:
    cfg = ResNetConfiguration()
    if name == "resnet18":
        # Use a resnet backbone.
        backbone = torchvision.models.resnet18()
        backbone.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Ignore the classification head as we only want the features.
        backbone.fc = torch.nn.Identity()
    elif name == "resnet50":
        # Use a resnet backbone.
        backbone = torchvision.models.resnet50()
        backbone.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Ignore the classification head as we only want the features.
        backbone.fc = torch.nn.Identity()        
        
        cfg.batch_size = 64
        cfg.dim = 2048

    elif name == "resnet101":
        # Use a resnet backbone.
        backbone = torchvision.models.resnet101()
        backbone.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Ignore the classification head as we only want the features.
        backbone.fc = torch.nn.Identity()        
        
        cfg.batch_size = 64
        cfg.dim = 2048
    else:
        raise NotImplementedError(f"`{name}` not implemented")
    return backbone, cfg
