
import os
import torch
import torchvision
from dataclasses import dataclass
from typing import Tuple

import sys
sys.path.insert(0, "../")
from DEFAULTS import BASE_PATH

class ConvNextWeights:

    CONVNEXT_TINY_IMAGENET1K_V1 = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    CONVNEXT_TINY_SSL_STED = os.path.join(BASE_PATH, "baselines", "convnext-tiny_STED", "result.pt")
    CONVNEXT_TINY_SSL_CTC = os.path.join(BASE_PATH, "baselines", "convnext-tiny_CTC", "result.pt")

    CONVNEXT_SMALL_IMAGENET1K_V1 = torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1
    CONVNEXT_SMALL_SSL_STED = os.path.join(BASE_PATH, "baselines", "convnext-small_STED", "result.pt")

    CONVNEXT_BASE_IMAGENET1K_V1 = torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1
    CONVNEXT_BASE_SSL_STED = os.path.join(BASE_PATH, "baselines", "convnext-base_STED", "result.pt")

    CONVNEXT_LARGE_IMAGENET1K_V1 = torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1
    CONVNEXT_LARGE_SSL_STED = os.path.join(BASE_PATH, "baselines", "convnext-large_STED", "result.pt")

@dataclass
class ConvNextConfiguration:
    
    freeze_backbone: bool = False
    backbone: str = "convnext"
    backbone_weights: str = None
    batch_size: int = 64
    dim: int = 768
    in_channels: int = 1

def get_backbone(name: str, **kwargs) -> Tuple[torch.nn.Module, ConvNextConfiguration]:
    cfg = ConvNextConfiguration()
    for key, value in kwargs.items():
        setattr(cfg, key, value)

    if name == "convnext-tiny":
        # Use a convnext backbone.
        backbone = torchvision.models.convnext_tiny()
        backbone.features[0][0] = torch.nn.Conv2d(in_channels=cfg.in_channels, out_channels=96, kernel_size=(4, 4), stride=(4, 4))

        # Ignore the classification head as we only want the features.
        backbone.classifier = torch.nn.Identity()

    elif name == "convnext-small":
        # Use a convnext backbone.
        backbone = torchvision.models.convnext_small()
        backbone.features[0][0] = torch.nn.Conv2d(in_channels=cfg.in_channels, out_channels=96, kernel_size=(4, 4), stride=(4, 4))

        # Ignore the classification head as we only want the features.
        backbone.classifier = torch.nn.Identity()   

        cfg.batch_size = 32

    elif name == "convnext-base":
        # Use a convnext backbone.
        backbone = torchvision.models.convnext_base()
        backbone.features[0][0] = torch.nn.Conv2d(in_channels=cfg.in_channels, out_channels=128, kernel_size=(4, 4), stride=(4, 4))

        # Ignore the classification head as we only want the features.
        backbone.classifier = torch.nn.Identity()

        cfg.dim = 1024
        cfg.batch_size = 16

    elif name == "convnext-large":
        # Use a convnext backbone.
        backbone = torchvision.models.convnext_large()
        backbone.features[0][0] = torch.nn.Conv2d(in_channels=cfg.in_channels, out_channels=192, kernel_size=(4, 4), stride=(4, 4))

        # Ignore the classification head as we only want the features.
        backbone.classifier = torch.nn.Identity()

        cfg.dim = 1536
        cfg.batch_size = 16        
    else:
        raise NotImplementedError(f"`{name}` not implemented")
    return backbone, cfg
