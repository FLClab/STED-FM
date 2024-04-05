import torch
import torchvision
from dataclasses import dataclass

@dataclass
class ResNetConfiguration:
    
    batch_size: int = 256

def get_backbone(name: str) -> torch.nn.Module:
    if name == "resnet18":
        # Use a resnet backbone.
        backbone = torchvision.models.resnet18()
        backbone.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Ignore the classification head as we only want the features.
        backbone.fc = torch.nn.Identity()
    else:
        raise NotImplementedError(f"`{name}` not implemented")
    return backbone, ResNetConfiguration()
