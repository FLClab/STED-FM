
import torch

from .resnet import get_backone as get_resnet_backbone
from .micranet import get_backone as get_micranet_backbone

BACKBONES = {
    "resnet18" : get_resnet_backbone,
    "micranet" : get_micranet_backbone
}

def get_backbone(name : str) -> torch.nn.Module:
    return BACKBONES[name](name)