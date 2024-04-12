
import torch

from .resnet import get_backbone as get_resnet_backbone
from .micranet import get_backbone as get_micranet_backbone
from .convnext import get_backbone as get_convnext_backbone
from .naive import get_backbone as get_naive_backbone
from .unet import get_backbone as get_unet_backbone

BACKBONES = {
    "resnet18" : get_resnet_backbone,
    "micranet" : get_micranet_backbone,
    "convnext" : get_convnext_backbone,
    "naive" : get_naive_backbone,
    "unet" : get_unet_backbone,
}

def get_backbone(name : str) -> torch.nn.Module:
    if not name in BACKBONES:
        raise NotImplementedError(f"`{name}` is not a valid option.")
    return BACKBONES[name](name)