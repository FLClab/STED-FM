
import os
import torch
import torchvision

from dataclasses import dataclass

import sys
sys.path.insert(0, "../")
from DEFAULTS import BASE_PATH
from configuration import Configuration

class ResNetWeights:

    RESNET18_IMAGENET1K_V1 = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    RESNET18_SSL_HPA = os.path.join(BASE_PATH, "baselines", "dataset-fullimages-1Msteps-multigpu", "resnet18_HPA", "result.pt")
    RESNET18_SSL_JUMP = os.path.join(BASE_PATH, "baselines", "dataset-fullimages-1Msteps-multigpu", "resnet18_JUMP", "result.pt")
    RESNET18_SSL_SIM = os.path.join(BASE_PATH, "baselines", "dataset-fullimages-1Msteps-multigpu", "resnet18_SIM", "result.pt")    
    RESNET18_SSL_STED = os.path.join(BASE_PATH, "baselines", "dataset-fullimages-1Msteps-multigpu", "resnet18_STED", "result.pt")

    RESNET50_IMAGENET1K_V1 = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    RESNET50_SSL_HPA = os.path.join(BASE_PATH, "baselines", "dataset-fullimages-1Msteps-multigpu", "resnet50_HPA", "result.pt")
    RESNET50_SSL_JUMP = os.path.join(BASE_PATH, "baselines", "dataset-fullimages-1Msteps-multigpu", "resnet50_JUMP", "result.pt")
    RESNET50_SSL_SIM = os.path.join(BASE_PATH, "baselines", "dataset-fullimages-1Msteps-multigpu", "resnet50_SIM", "result.pt")
    RESNET50_SSL_STED = os.path.join(BASE_PATH, "baselines", "dataset-fullimages-1Msteps-multigpu", "resnet50_STED", "result.pt")

    RESNET101_IMAGENET1K_V1 = torchvision.models.ResNet101_Weights.IMAGENET1K_V1
    RESNET101_SSL_HPA = os.path.join(BASE_PATH, "baselines", "dataset-fullimages-1Msteps-multigpu", "resnet101_HPA", "result.pt")
    RESNET101_SSL_JUMP = os.path.join(BASE_PATH, "baselines", "dataset-fullimages-1Msteps-multigpu", "resnet101_JUMP", "result.pt")
    RESNET101_SSL_SIM = os.path.join(BASE_PATH, "baselines", "dataset-fullimages-1Msteps-multigpu", "resnet101_SIM", "result.pt")    
    RESNET101_SSL_STED = os.path.join(BASE_PATH, "baselines", "dataset-fullimages-1Msteps-multigpu", "resnet101_STED", "result.pt")

class ResNetConfiguration(Configuration):
    
    backbone: str = "resnet"
    backbone_weights: str = None
    batch_size: int = 256
    dim: int = 512
    freeze_backbone: bool = False
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
        cfg.batch_size = 256
        cfg.dim = 512

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
    elif name == "resnet152": 

        # Use a resnet backbone.
        backbone = torchvision.models.resnet152()
        backbone.conv1 = torch.nn.Conv2d(in_channels=cfg.in_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Ignore the classification head as we only want the features.
        backbone.fc = torch.nn.Identity()        
        
        cfg.batch_size = 64
        cfg.dim = 2048        
    else:
        raise NotImplementedError(f"`{name}` not implemented")
    return backbone, cfg
