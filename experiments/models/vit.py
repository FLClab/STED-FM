
import torch
import torchvision
from dataclasses import dataclass

from timm.models.vision_transformer import vit_small_patch16_224, vit_base_patch16_224

@dataclass
class ViTConfiguration:
    
    backbone: str = "vit"
    batch_size: int = 256
    dim: int = 384

def get_backbone(name: str) -> torch.nn.Module:
    cfg = ViTConfiguration()
    print(f"--- {name} ---")
    if name == "vit-small":
        # Use a vit-small backbone.
        backbone = vit_small_patch16_224(in_chans=1)
    elif name == "vit-base":
        # Use a vit-base backbone.
        backbone = vit_base_patch16_224(in_chans=1)     
    else:
        raise NotImplementedError(f"`{name}` not implemented")
    return backbone, cfg
