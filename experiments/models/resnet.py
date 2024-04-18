
import os
import torch
import torchvision

from dataclasses import dataclass

import sys
sys.path.insert(0, "../")
from DEFAULTS import BASE_PATH

class ResNetWeights:

    RESNET18_IMAGENET1K_V1 = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    RESNET18_SSL_STED = os.path.join(BASE_PATH, "baselines", "resnet18", "result.pt")
    RESNET18_SSL_CTC = os.path.join(BASE_PATH, "baselines", "resnet18_CTC", "result.pt")

    RESNET50_IMAGENET1K_V1 = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    RESNET50_SSL_STED = os.path.join(BASE_PATH, "baselines", "resnet50", "result.pt")
    RESNET50_SSL_CTC = os.path.join(BASE_PATH, "baselines", "resnet50_CTC", "result.pt")

    RESNET101_IMAGENET1K_V1 = torchvision.models.ResNet101_Weights.IMAGENET1K_V1
    RESNET101_SSL_STED = os.path.join(BASE_PATH, "baselines", "resnet101", "result.pt")

@dataclass
class ResNetConfiguration:
    
    backbone: str = "resnet"
    batch_size: int = 256
    dim: int = 512
    in_channels: int = 1

def get_backbone(name: str, **kwargs) -> torch.nn.Module:
    cfg = ResNetConfiguration()
    for key, value in kwargs.items():
        setattr(cfg, key, value)

    if name == "resnet18":
        # Use a resnet backbone.
        backbone = torchvision.models.resnet18()
        backbone.conv1 = torch.nn.Conv2d(in_channels=cfg.in_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Ignore the classification head as we only want the features.
        backbone.fc = torch.nn.Identity()

    elif name == "resnet50":
        # Use a resnet backbone.
        backbone = torchvision.models.resnet50()
        backbone.conv1 = torch.nn.Conv2d(in_channels=cfg.in_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Ignore the classification head as we only want the features.
        backbone.fc = torch.nn.Identity()        
        
        cfg.batch_size = 64
        cfg.dim = 2048

    elif name == "resnet101":
        # Use a resnet backbone.
        backbone = torchvision.models.resnet101()
        backbone.conv1 = torch.nn.Conv2d(in_channels=cfg.in_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Ignore the classification head as we only want the features.
        backbone.fc = torch.nn.Identity()        
        
        cfg.batch_size = 64
        cfg.dim = 2048
    else:
        raise NotImplementedError(f"`{name}` not implemented")
    return backbone, cfg
