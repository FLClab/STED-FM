import torch
import os

from typing import Union, Any
from enum import Enum
from collections import OrderedDict
from torch.hub import load_state_dict_from_url

from .resnet import ResNetWeights
from .micranet import MICRANetWeights
from .convnext import ConvNextWeights
from .lightly_mae import MAEWeights

MODELS = {
    "resnet18" : ResNetWeights,
    "resnet50" : ResNetWeights,
    "resnet101" : ResNetWeights,
    "micranet" : MICRANetWeights,
    "convnext-tiny" : ConvNextWeights,
    "convnext-small" : ConvNextWeights,
    "convnext-base" : ConvNextWeights,
    "convnext-large": ConvNextWeights,
    'vit-small': None,
    'mae-tiny': MAEWeights,
    'mae': MAEWeights, # mae defaults to mae-small
    'mae-small': MAEWeights,
    'mae-lightning-tiny': MAEWeights,
    'mae-lightning-small': MAEWeights,
    'mae-lightning-base': MAEWeights,
    'mae-lightning-large': MAEWeights,
    'mae-base': MAEWeights,
    'mae-large': MAEWeights
}

def handle_url_state_dict(name: str, weights: Union[str, Enum]) -> dict:
    if "mae" in name.lower():
        state_dict = load_state_dict_from_url(weights.url, map_location="cpu")
        return state_dict
    elif "convnext" in name.lower():
        state_dict = load_state_dict_from_url(weights.url, map_location="cpu")
        state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k}
    else:
        state_dict = load_state_dict_from_url(weights.url, map_location="cpu")
        print(f"--- {name} | {weights} ---\n")
        state_dict = {k: v for k, v in state_dict.items() if "fc" not in k}
        return state_dict

def handle_str_state_dict(name: str, weights: Union[str, Enum]) -> dict:
    state_dict = torch.load(weights, map_location="cpu")

    if "mae" in name.lower():
        return state_dict["state_dict"] 
    
    elif "micranet" in name.lower():
        return state_dict["state_dict"]["backbone"]

    elif "resnet" in name.lower():
        if "hpa" in weights.lower():
            return state_dict["state_dict"]["backbone"]
        elif "sted" in weights.lower():     
            return {key.replace("backbone.", ""): values for key, values in state_dict["state_dict"].items() if "backbone" in key}

    elif "convnext" in name.lower():
        return state_dict["state_dict"]["backbone"]

    else:
        raise NotImplementedError(f"Weights `{weights}` not supported.")

def load_weights(name: str, weights: Union[str, Enum, None]) -> dict:
    # TODO: Fix all the if branches that are currently here to satisfy the strict=true loading condition after returning
    # Most probably weights from pretrained torchvision models

    if isinstance(weights, Enum):
        print(f"--- {name} | Pretrained Image-Net ---\n")
        state_dict = handle_url_state_dict(name, weights=weights)
        return state_dict
    elif isinstance(weights, str):
        print(f"--- {name} | {weights} ---\n")
        state_dict = handle_str_state_dict(name=name, weights=weights)
        return state_dict

        # state_dict = torch.load(weights, map_location="cpu")
        # if "model" in state_dict:
        #     print(f"\t`model` state dict: {state_dict.keys()}")
        #     # Model pretrained using SimCLR
        #     try:
        #         return state_dict["model"]["backbone"]
        #     except KeyError:
        #         return state_dict['model']
        # elif "model_state_dict" in state_dict:
        #     print(f"\t`model_state_dict` state dict: {state_dict.keys()}")
        #     return state_dict["model_state_dict"]
        # elif "state_dict" in state_dict:
        #     print(f"\t`state_dict` state dict: {state_dict.keys()}")
        #     # print(type(state_dict['state_dict']), state_dict['state_dict'].keys())
        #     if "resnet" in weights.lower():
        #         print("Editing state dict keys for ResNet...")
        #         state_dict = {key.replace("backbone.", ""): values for key, values in state_dict["state_dict"].items() if "backbone" in key}
        #         state_dict = {k: v for k, v in state_dict.items() if "fc" not in k}
        #         return state_dict 
        #     else:
        #         return state_dict['state_dict']
        # else:
        #     raise KeyError(f"No model found in checkpoint.") 
        
    elif weights is None:
        print(f"--- {name} | Pretrained Image-Net ---\n")
        return None
    else:
        raise NotImplementedError("Weights not implemented yet.")

def get_weights(name : str, weights: Union[str, None]) -> torch.nn.Module:
    if weights is None:
        return None
    if not name in MODELS:
        raise NotImplementedError(f"`{name}` is not a valid option.")
    
    if os.path.isfile(weights):
        return load_weights(name, weights)
    
    weights = getattr(MODELS[name], weights)
    return load_weights(name, weights)