import torch
import os
import zipfile

from typing import Union, Any
from enum import Enum
from collections import OrderedDict
from torch.hub import load_state_dict_from_url, download_url_to_file
from urllib.parse import urlparse

from .resnet import ResNetWeights
from .micranet import MICRANetWeights
from .convnext import ConvNextWeights
from .lightly_mae import MAEWeights

from stedfm.DEFAULTS import BASE_PATH

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
    'mae-large': MAEWeights,
    "mae-lightning-64-p8": MAEWeights,
    "mae-64-p8": MAEWeights,
    "mae-lightning-224-p16": MAEWeights,
    "mae-224-p16": MAEWeights,
}

def get_state_dict(name: str, state_dict: dict) -> dict:
    if "mae" in name.lower():
        return state_dict["state_dict"] 
    
    elif "micranet" in name.lower():
        return state_dict["state_dict"]["backbone"]

    elif "resnet" in name.lower():
        if "simclr" in state_dict["hyper_parameters"]["cfg"]:
            print("SimCLR weights detected.")
            return {key.replace("backbone.", ""): values for key, values in state_dict["state_dict"].items() if "backbone" in key}
        elif "dino" in state_dict["hyper_parameters"]["cfg"]:
            print("DINO weights detected.")
            return {key.replace("student_backbone.", ""): values for key, values in state_dict["state_dict"].items() if "student_backbone" in key}
        else:
            raise NotImplementedError(f"Weights not implemented yet for this resnet model.")
    elif "convnext" in name.lower():
        return {key.replace("backbone.", ""): values for key, values in state_dict["state_dict"].items() if "backbone" in key}
        # return state_dict["state_dict"]["backbone"]
    else:
        raise NotImplementedError(f"Weights not supported.")

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

    return get_state_dict(name, state_dict)

def handle_url_zip_state_dict(name: str, weights: str) -> dict:

    url = weights

    model_dir = BASE_PATH
    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        print(f"Downloading {filename} to {cached_file} ...")
        download_url_to_file(url, cached_file, progress=True)

    # Handle the zipfile
    with zipfile.ZipFile(cached_file) as f:
        members = f.infolist()
        extracted = False
        for member in members:
            if member.filename.endswith('.pth'):
                extracted_file = os.path.join(model_dir, member.filename)
                if os.path.exists(extracted_file):
                    extracted = True
                    break
        if not extracted:
            print(f"Extracting {filename} to {model_dir} ...")
            f.extractall(model_dir)

    print(f"Loading state_dict from {extracted_file}")
    state_dict = torch.load(extracted_file, map_location="cpu")
    return get_state_dict(name, state_dict)

def load_weights(name: str, weights: Union[str, Enum, None]) -> dict:
    if isinstance(weights, Enum):
        print(f"--- {name} | Pretrained Image-Net ---\n")
        state_dict = handle_url_state_dict(name, weights=weights)
        return state_dict
    elif isinstance(weights, str):
        if "http" in weights or "https" in weights:
            if weights.endswith(".zip"):
                print(f"--- {name} | ({weights}) Downloading from URL and extracting zip ---\n")
                state_dict = handle_url_zip_state_dict(name, weights)
            else:
                print(f"--- {name} | ({weights}) Downloading from URL ---\n")
                state_dict = handle_url_state_dict(name, weights=weights)
                return state_dict
        elif os.path.isfile(weights):
            print(f"--- {name} | ({weights}) Loading from local file ---\n")
            state_dict = handle_str_state_dict(name, weights=weights)
            return state_dict
        else:
            raise NotImplementedError(f"Invalid weights path: {weights}. It should be a URL or a local file path.")
    elif weights is None:
        print(f"--- {name} | Pretrained Image-Net or from scratch ---\n")
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