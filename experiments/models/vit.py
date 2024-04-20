
import torch
import torchvision
from dataclasses import dataclass

from timm.models.vision_transformer import vit_small_patch16_224, vit_base_patch16_224

class ViTWeights:
    VIT_SUPERVISED_PROTEINS = "/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/MAE_fully-supervised/synaptic-proteins/vit-small_from-scratch_model.pth"
    VIT_SUPERVISED_OPTIM = "/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/MAE_fully-supervised/optim/vit-small_from-scratch_model.pth"


@dataclass
class ViTConfiguration:
    
    backbone: str = "vit-small"
    batch_size: int = 256
    dim: int = 384
    in_channels: int = 1

def get_backbone(name: str, **kwargs) -> torch.nn.Module:
    cfg = ViTConfiguration()
    for key, value in kwargs.items():
        setattr(cfg, key, value)

    print(f"--- {name} ---")
    if name == "vit-small":
        # Use a vit-small backbone.
        backbone = vit_small_patch16_224(in_chans=cfg.in_channels)
    elif name == "vit-base":
        # Use a vit-base backbone.
        backbone = vit_base_patch16_224(in_chans=cfg.in_channels)     
    else:
        raise NotImplementedError(f"`{name}` not implemented")
    return backbone, cfg
