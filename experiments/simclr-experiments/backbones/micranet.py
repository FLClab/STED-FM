
import torch
import torchvision

def get_backbone(name: str) -> torch.nn.Module:
    raise NotImplementedError(f"`{name}` not implemented")
    return backbone
