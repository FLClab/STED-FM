
import os
import torch
import torchvision

from dataclasses import dataclass

import sys
sys.path.insert(0, "../")
from DEFAULTS import BASE_PATH
from configuration import Configuration

class ResNetWeights:

    RESNET18_IMAGENET1K_V1 = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    # RESNET18_SSL_HPA = os.path.join(BASE_PATH, "baselines", "resnet18_HPA", "result.pt")
    RESNET18_SSL_HPA = os.path.join(BASE_PATH, "baselines", "dataset-fullimages-1Msteps-multigpu", "resnet18_HPA", "result.pt")
    RESNET18_SSL_JUMP = os.path.join(BASE_PATH, "baselines", "dataset-fullimages-1Msteps-multigpu", "resnet18_JUMP", "result.pt")
    # RESNET18_SSL_STED = os.path.join(BASE_PATH, "baselines", "resnet18_STED", "result.pt")
    # RESNET18_SSL_STED = os.path.join(BASE_PATH, "baselines", "tests", "dataset-fullimages-200epochs-singlegpu/resnet18_STED", "result.pt")
    # RESNET18_SSL_STED = os.path.join("/home/anbil106/projects/def-flavielc", "baselines", "resnet18_STED", "result.pt")
    
    RESNET18_SSL_STED = os.path.join(BASE_PATH, "baselines", "dataset-crops-1Msteps-multigpu", "resnet18_STED", "result.pt")
    RESNET18_SSL_CTC = os.path.join(BASE_PATH, "baselines", "resnet18_CTC", "result.pt")

    RESNET50_IMAGENET1K_V1 = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    RESNET50_SSL_HPA = os.path.join(BASE_PATH, "baselines", "dataset-fullimages-1Msteps-multigpu", "resnet50_HPA", "result.pt")
    RESNET50_SSL_JUMP = os.path.join(BASE_PATH, "baselines", "dataset-fullimages-1Msteps-multigpu", "resnet50_JUMP", "result.pt")
    RESNET50_SSL_STED = os.path.join(BASE_PATH, "baselines", "dataset-fullimages-1Msteps-multigpu", "resnet50_STED", "result.pt")
    # RESNET50_SSL_STED = os.path.join(BASE_PATH, "baselines", "dataset-fullimages-1Msteps-multigpu/resnet50_STED", "result.pt")
    # RESNET50_SSL_STED = os.path.join(BASE_PATH, "baselines", "dataset-fullimages-1Msteps-multigpu/resnet50_STED", "checkpoint-145000.pt")
    RESNET50_SSL_CTC = os.path.join(BASE_PATH, "baselines", "resnet50_CTC", "result.pt")


    # RESNET18_LINEARPROBE_IMAGENET_PROTEINS = None
    # RESNET18_LINEARPROBE_CTC_PROTEINS = None
    # RESNET18_LINEARPROBE_STED_PROTEINS = None 
    # RESNET18_LINEARPROBE_IMAGENET_OPTIM = None
    # RESNET18_LINEARPROBE_CTC_OPTIM = None
    # RESNET18_LINEARPROBE_STED_OPTIM = None     

    # RESNET18_LINEARPROBE_IMAGENET_PROTEINS = "/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/resnet18_ImageNet/optim/finetuned_4blocks_model.pth"
    # RESNET18_LINEARPROBE_CTC_PROTEINS = "/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/resnet18_CTC/optim/finetuned_4blocks_model.pth"
    # RESNET18_LINEARPROBE_STED_PROTEINS = "/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/resnet18_STED/synaptic-proteins/finetuned_4blocks_model.pth" 
    # RESNET18_LINEARPROBE_IMAGENET_OPTIM = "/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/resnet18_ImageNet/optim/finetuned_4blocks_model.pth"
    # RESNET18_LINEARPROBE_CTC_OPTIM = "/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/resnet18_CTC/optim/finetuned_4blocks_model.pth"
    # RESNET18_LINEARPROBE_STED_OPTIM = "/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/resnet18_STED/optim/finetuned_4blocks_model.pth" 


class ResNetConfiguration(Configuration):
    
    backbone: str = "resnet"
    backbone_weights: str = None
    batch_size: int = 256
    dim: int = 512
    freeze_backbone: bool = False
    in_channels: int = 1


def get_backbone(name: str, **kwargs) -> torch.nn.Module:
    cfg = ResNetConfiguration()
    for key, value in kwargs.items():
        setattr(cfg, key, value)

    if name == "resnet18":
        # Use a resnet backbone.
        backbone = torchvision.models.resnet18()
        backbone.conv1 = torch.nn.Conv2d(in_channels=cfg.in_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Ignore the classification head as we only want the features.
        backbone.fc = torch.nn.Identity()
        cfg.batch_size = 256
        cfg.dim = 512

    elif name == "resnet50":
        # Use a resnet backbone.
        backbone = torchvision.models.resnet50()
        backbone.conv1 = torch.nn.Conv2d(in_channels=cfg.in_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Ignore the classification head as we only want the features.
        backbone.fc = torch.nn.Identity()        
        
        cfg.batch_size = 64
        cfg.dim = 2048

    elif name == "resnet101":
        # Use a resnet backbone.
        backbone = torchvision.models.resnet101()
        backbone.conv1 = torch.nn.Conv2d(in_channels=cfg.in_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Ignore the classification head as we only want the features.
        backbone.fc = torch.nn.Identity()        
        
        cfg.batch_size = 64
        cfg.dim = 2048
    elif name == "resnet152": 

        # Use a resnet backbone.
        backbone = torchvision.models.resnet152()
        backbone.conv1 = torch.nn.Conv2d(in_channels=cfg.in_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Ignore the classification head as we only want the features.
        backbone.fc = torch.nn.Identity()        
        
        cfg.batch_size = 64
        cfg.dim = 2048        
    else:
        raise NotImplementedError(f"`{name}` not implemented")
    return backbone, cfg
