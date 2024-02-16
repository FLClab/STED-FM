
import torch

from .resnet import get_backbone as get_resnet_backbone
from .micranet import get_backbone as get_micranet_backbone

BACKBONES = {
    "resnet18" : get_resnet_backbone,
    "micranet" : get_micranet_backbone
}

def get_backbone(name : str) -> torch.nn.Module:
    if not name in BACKBONES:
        raise NotImplementedError(f"`{name}` is not a valid option.")
    return BACKBONES[name](name)