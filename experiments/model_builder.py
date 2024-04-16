from models.lightly_mae import LightlyMAE
from models.classifier import LinearProbe
from timm.models.vision_transformer import vit_small_patch16_224
import lightly.models.utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
import torch
import torchvision

from models import get_model
from models.loading import get_weights

def get_base_model(name: str, **kwargs):
    model, cfg = get_model(name, **kwargs)
    return model, cfg

def get_pretrained_model(name: str, weights: str = None, path: str = None, **kwargs):
    if name == "MAE":
        if weights == "ImageNet":
            vit = vit_small_patch16_224(in_chans=3, pretrained=True)
            model = LightlyMAE(vit=vit, in_channels=3, mask_ratio=0.0)
        elif weights == "CTC":
            vit = vit_small_patch16_224(in_chans=1)
            model = LightlyMAE(vit=vit, in_channels=1, mask_ratio=0.0)
            checkpoint = torch.load("/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/Cell-Tracking-Challenge/baselines/checkpoint-530.pth")
            model.load_state_dict(checkpoint['model'])
        elif weights == "STED":
            vit = vit_small_patch16_224(in_chans=1)
            model = LightlyMAE(vit=vit, in_channels=1, mask_ratio=0.0)
            checkpoint = torch.load("/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/checkpoint-530.pth")
            model.load_state_dict(checkpoint['model'])
        else:
            raise NotImplementedError(f"Weights {weights} not supported yet for model {name}.")
    elif name == "MAEClassifier":
        if weights == "ImageNet":
            print("-- Loading ImageNet ViT ---")
            vit = vit_small_patch16_224(in_chans=3, pretrained=True)
            backbone = LightlyMAE(vit=vit, in_channels=3, mask_ratio=0.0)
            # No need to load any checkpoint into the full MAE because the Decoder is never used in fine-tuning
            # So only need to load the encoder checkpoint (pretrained weights)
            model = LinearProbe(
                backbone=backbone, 
                name="MAE",
                num_classes=4,
                freeze=kwargs['freeze'],
                global_pool='avg'
            )
        elif weights == "CTC":
            print("-- Loading CTC ViT ---")
            vit = vit_small_patch16_224(in_chans=1)
            backbone = LightlyMAE(vit=vit, in_channels=1, mask_ratio=0.0)
            checkpoint = torch.load("/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/Cell-Tracking-Challenge/baselines/checkpoint-530.pth")
            backbone.load_state_dict(checkpoint['model'])
            model = LinearProbe(
                backbone=backbone,
                name="MAE",
                num_classes=4,
                freeze=kwargs['freeze'],
                global_pool="avg"
            )
        elif weights == "STED":
            print("--- Loading STED ViT ---")
            vit = vit_small_patch16_224(in_chans=1)
            backbone = LightlyMAE(vit=vit, in_channels=1, mask_ratio=0.0)
            checkpoint = torch.load("/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/checkpoint-530.pth")
            backbone.load_state_dict(checkpoint['model'])
            model = LinearProbe(
                backbone=backbone,
                name="MAE",
                num_classes=4,
                freeze=kwargs['freeze'],
                global_pool="avg"
            )
        else:
            raise NotImplementedError(f"Weights {weights} not supported yet for model {name}.")
    elif name in ["resnet18", "resnet50", "resnet101", "micranet", "convnext-tiny", "convnext-small", "convnext-base"]:
        model, cfg = get_base_model(name, in_channels=3 if "imagenet" in weights.lower() else 1)
        state_dict = get_weights(name, weights)
        model.load_state_dict(state_dict, strict=False)
        return model, cfg
    else:
        raise NotImplementedError(f"Model {name} not implemented yet.")
    return model


def get_classifier(name: str, pretraining: str, task:str, path: str = None):
    if name == "MAE":
        if pretraining == "ImageNet":
            print("--- Loading ImageNet ViT fine-tuned ---")
            vit = vit_small_patch16_224(in_chans=3, pretrained=True)
            backbone = LightlyMAE(vit=vit, in_channels=3, mask_ratio=0.0)
            model = LinearProbe(
                backbone=backbone,
                name="MAE",
                num_classes=4,
                freeze=True,
                global_pool='avg'
            )
            if path is not None:
                checkpoint = torch.load(f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/finetuning/ImageNet/{task}_{path}_model.pth")
            else:
                checkpoint = torch.load(f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/finetuning/ImageNet/{task}_model.pth")
            model.load_state_dict(checkpoint["model_state_dict"])
            return model
        elif pretraining == "CTC":
            print("-- Loading CTC ViT fine-tuned---")
            vit = vit_small_patch16_224(in_chans=1)
            backbone = LightlyMAE(vit=vit, in_channels=1, mask_ratio=0.0)
            model = LinearProbe(
                backbone=backbone,
                name="MAE",
                num_classes=4,
                freeze=True,
                global_pool="avg"
            )
            if path is not None:
                checkpoint = torch.load(f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/finetuning/CTC/{task}_{path}_model.pth")
            else:
                checkpoint = torch.load(f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/finetuning/CTC/{task}_model.pth")
            model.load_state_dict(checkpoint['model_state_dict'])
            return model
        elif pretraining == "STED":
            print("--- Loading STED ViT ---")
            vit = vit_small_patch16_224(in_chans=1)
            backbone = LightlyMAE(vit=vit, in_channels=1, mask_ratio=0.0)
            model = LinearProbe(
                backbone=backbone,
                name="MAE",
                num_classes=4,
                freeze=True,
                global_pool="avg"
            )
            if path is not None:
                checkpoint = torch.load(f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/finetuning/STED/{task}_{path}_model.pth")
            else:
                checkpoint = torch.load(f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/finetuning/STED/{task}_model.pth")
            model.load_state_dict(checkpoint['model_state_dict'])
            return model
        else:
            raise NotImplementedError(f"Pretraining {pretraining} not supported.")
    else: 
        raise not NotImplementedError(f"Model {name} not implemented yet.")
