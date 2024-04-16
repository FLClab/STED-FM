
import torch
import torchvision
from dataclasses import dataclass

@dataclass
class ConvNextConfiguration:
    
    backbone: str = "convnext"
    batch_size: int = 64
    dim: int = 768

def get_backbone(name: str) -> tuple[torch.nn.Module, ConvNextConfiguration]:
    cfg = ConvNextConfiguration()
    if name == "convnext-tiny":
        # Use a convnext backbone.
        backbone = torchvision.models.convnext_tiny()
        backbone.features[0][0] = torch.nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(4, 4), stride=(4, 4))

        # Ignore the classification head as we only want the features.
        backbone.classifier = torch.nn.Identity()

    elif name == "convnext-small":
        # Use a convnext backbone.
        backbone = torchvision.models.convnext_small()
        backbone.features[0][0] = torch.nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(4, 4), stride=(4, 4))

        # Ignore the classification head as we only want the features.
        backbone.classifier = torch.nn.Identity()   

    elif name == "convnext-base":
        # Use a convnext backbone.
        backbone = torchvision.models.convnext_base()
        backbone.features[0][0] = torch.nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4, 4), stride=(4, 4))

        # Ignore the classification head as we only want the features.
        backbone.classifier = torch.nn.Identity()

        cfg.dim = 1024
    else:
        raise NotImplementedError(f"`{name}` not implemented")
    return backbone, cfg
