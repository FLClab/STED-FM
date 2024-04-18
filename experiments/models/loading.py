import torch

from typing import Union, Any
from enum import Enum
from collections import OrderedDict
from torch.hub import load_state_dict_from_url

from .resnet import ResNetWeights
from .micranet import MICRANetWeights
from .convnext import ConvNextWeights

MODELS = {
    "resnet18" : ResNetWeights,
    "resnet50" : ResNetWeights,
    "resnet101" : ResNetWeights,
    "micranet" : MICRANetWeights,
    "convnext-tiny" : ConvNextWeights,
    "convnext-small" : ConvNextWeights,
    "convnext-base" : ConvNextWeights,
    'vit-small': None,
    'mae': None,
}

def load_weights(weights: Union[str, Enum]) -> dict:
    # Most probably weights from pretrained torchvision models
    if isinstance(weights, Enum):
        return load_state_dict_from_url(weights.url, map_location="cpu")
    elif isinstance(weights, str):
        state_dict = torch.load(weights, map_location="cpu")
        if "model" in state_dict:
            # Model pretrained using SimCLR
            return state_dict["model"]["backbone"]
        return state_dict
    else:
        raise NotImplementedError("Weights not implemented yet.")

def get_weights(name : str, weights: str) -> torch.nn.Module:
    if not name in MODELS:
        raise NotImplementedError(f"`{name}` is not a valid option.")
    weights = getattr(MODELS[name], weights)
    return load_weights(weights)