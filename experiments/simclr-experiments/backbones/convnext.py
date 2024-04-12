
import torch
import torchvision
from dataclasses import dataclass

@dataclass
class ConvNextConfiguration:
    
    batch_size: int = 64
    dim: int = 768

def get_backbone(name: str) -> torch.nn.Module:
    cfg = ConvNextConfiguration()
    if name == "convnext":
        # Use a resnet backbone.
        backbone = torchvision.models.convnext_tiny()
        backbone.features[0][0] = torch.nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(4, 4), stride=(4, 4))

        # Ignore the classification head as we only want the features.
        backbone.classifier = torch.nn.Identity()
    else:
        raise NotImplementedError(f"`{name}` not implemented")
    return backbone, cfg
