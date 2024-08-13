
import torch 
from dataclasses import dataclass   

from .unet import get_decoder as get_unet_decoder
from .vit import get_decoder as get_vit_decoder

MODELS = {
    "resnet" : get_unet_decoder,
    "resnet18" : get_unet_decoder,
    "resnet50" : get_unet_decoder,
    "resnet101" : get_unet_decoder,
    "micranet" : get_unet_decoder,
    "convnext" : get_unet_decoder,
    "convenext-tiny" : get_unet_decoder,
    "convenext-small" : get_unet_decoder,
    "convenext-base" : get_unet_decoder,
    "vit-tiny" : get_vit_decoder,
    "vit-small" : get_vit_decoder,
    "vit-base" : get_vit_decoder,
    "mae-small" : get_vit_decoder,
    "mae-base" : get_vit_decoder,
}

def get_decoder(backbone: torch.nn.Module, cfg: dataclass, **kwargs) -> torch.nn.Module:
    """
    Creates the decoder based on the configuration

    :param backbone: A `torch.nn.Module` instance
    :param cfg: A `dataclass` instance

    :returns : A `torch.nn.Module` instance
    """
    if cfg.backbone not in MODELS:
        raise ValueError(f"Backbone {cfg.backbone} not supported")
    return MODELS[cfg.backbone](backbone, cfg, **kwargs)