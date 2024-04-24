
import torch

from .resnet import get_backbone as get_resnet_backbone
from .micranet import get_backbone as get_micranet_backbone
from .convnext import get_backbone as get_convnext_backbone
from .naive import get_backbone as get_naive_backbone
from .vit import get_backbone as get_vit_backbone
from .lightly_mae import get_backbone as get_mae_backbone

MODELS = {
    "resnet18" : get_resnet_backbone,
    "resnet50" : get_resnet_backbone,
    "resnet101" : get_resnet_backbone,
    "micranet" : get_micranet_backbone,
    "convnext-tiny" : get_convnext_backbone,
    "convnext-small" : get_convnext_backbone,
    "convnext-base" : get_convnext_backbone,
    "naive" : get_naive_backbone,
    "vit-small" : get_vit_backbone,
    "vit-base": get_vit_backbone,
    'mae-tiny': get_mae_backbone,
    "mae": get_mae_backbone,
    "mae-small": get_mae_backbone,
    'mae-base': get_mae_backbone,
    'mae-large': get_mae_backbone
}

CLASSIFIERS = {
    "resnet18" : None,
    "resnet50" : None,
    "resnet101" : None,
    "micranet" : None,
    "convnext-tiny" : None,
    "convnext-small" : None,
    "convnext-base" : None,
    "naive" : None,
    "vit-small" : None,
}

def get_model(name : str, **kwargs) -> torch.nn.Module:
    if not name in MODELS:
        raise NotImplementedError(f"`{name}` is not a valid option.")
    return MODELS[name](name, **kwargs)