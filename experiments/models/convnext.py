
import os
import torch
import torchvision
from dataclasses import dataclass

import sys
sys.path.insert(0, "../")
from DEFAULTS import BASE_PATH

class ConvNextWeights:

    CONVNEXT_TINY_IMAGENET1K_V1 = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
<<<<<<< HEAD
    CONVNEXT_TINY_SSL_STED = os.path.join(BASE_PATH, "baselines", "convnext-tiny", "result.pt")
    CONVNEXT_TINY_SSL_CTC = os.path.join(BASE_PATH, "baselines", "convnext-tiny_CTC", "result.pt")
=======
    CONVNEXT_SMALL_IMAGENET1K_V1 = torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1
    CONVNEXT_TINY_SSL_CTC = "/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/convnext-tiny_CTC/result.pt"
    CONVNEXT_TINY_SSL_STED = "/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/convnext-tiny_STED/result.pt"
>>>>>>> 070162485de1449e9cc061f9f95e89f7c29fe901

    CONVNEXT_SMALL_IMAGENET1K_V1 = torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1

    CONVNEXT_BASE_IMAGENET1K_V1 = torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1

@dataclass
class ConvNextConfiguration:
    
    backbone: str = "convnext"
    batch_size: int = 64
    dim: int = 768
    in_channels: int = 1

def get_backbone(name: str, **kwargs) -> tuple[torch.nn.Module, ConvNextConfiguration]:
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

    elif name == "convnext-base":
        # Use a convnext backbone.
        backbone = torchvision.models.convnext_base()
        backbone.features[0][0] = torch.nn.Conv2d(in_channels=cfg.in_channels, out_channels=128, kernel_size=(4, 4), stride=(4, 4))

        # Ignore the classification head as we only want the features.
        backbone.classifier = torch.nn.Identity()

        cfg.dim = 1024
    else:
        raise NotImplementedError(f"`{name}` not implemented")
    return backbone, cfg
