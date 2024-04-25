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


def get_pretrained_model_v2(name: str, weights: str = None, as_classifier: bool = False, path: str = None, **kwargs):
    if name in ["resnet18", "resnet50", "resnet101", "micranet", "convnext-tiny", "convnext-small", "convnext-base", "vit-small", "mae", "mae-tiny", "mae-small", "mae-base"]:
        backbone, cfg = get_base_model(name, **kwargs)
        state_dict = get_weights(name, weights)
        # This is could lead to errors if the model is not exactly the same as the one used for pretraining
        if state_dict is not None:
            print(f"--- Loading from state dict ---")
            backbone.load_state_dict(state_dict, strict=False)
        print(f"--- Loaded model {name} with weights {weights} ---")
        if as_classifier:
            model = LinearProbe(
                backbone=backbone,
                name=name,
                num_classes=4,
                num_blocks=kwargs['blocks'],
            )
            print(f"--- Added linear probe to {kwargs['blocks']} frozen blocks ---")
            return model, cfg
        else:
            return backbone, cfg
    else:
        raise NotImplementedError(f"Model {name} not implemented yet.")

def get_pretrained_model(name: str, weights: str = None, path: str = None, **kwargs):
    print("Query: ", name, weights)
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
            checkpoint = torch.load("/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/MAE_STED/checkpoint-530.pth")
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
                num_blocks=kwargs['blocks'],
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
                num_blocks=kwargs['blocks'],
                global_pool="avg"
            )
        elif weights == "STED":
            print("--- Loading STED ViT ---")
            vit = vit_small_patch16_224(in_chans=1)
            backbone = LightlyMAE(vit=vit, in_channels=1, mask_ratio=0.0)
            checkpoint = torch.load("/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/MAE_STED/checkpoint-530.pth")
            backbone.load_state_dict(checkpoint['model'])
            model = LinearProbe(
                backbone=backbone,
                name="MAE",
                num_classes=4,
                num_blocks=kwargs['blocks'],
                global_pool="avg"
            )
        else:
            raise NotImplementedError(f"Weights {weights} not supported yet for model {name}.")

    # elif name == "resnet18":
    #     if weights == "ImageNet":
    #         print("--- Loading ImageNet ResNet-18 ---")
    #         backbone = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    #         backbone.fc = torch.nn.Identity()
    #         model = LinearProbe(
    #             backbone=backbone, 
    #             name="resnet18",
    #             num_classes=4,
    #             freeze=kwargs['freeze'],
    #             global_pool=None,
    #         )
    #     elif weights == "CTC":
    #         raise NotImplementedError
    #     elif weights == "STED":
    #         print("Loading STED ResNet-18")
    #         backbone = torchvision.models.resnet18(weights=None)
    #         backbone.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #         backbone.fc = torch.nn.Identity()
    #         checkpoint = torch.load("/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/resnet18_STED/result.pt")
    #         backbone.load_state_dict(checkpoint['model']['backbone'])
    #         model = LinearProbe(
    #             backbone=backbone,
    #             name='resnet18',
    #             num_classes=4,
    #             freeze=kwargs['freeze'],
    #             global_pool=None
    #         )
    #     else:
    #         raise NotImplementedError(f"Weights {weights} not supported yet for model {name}.")

    # elif name == "resnet50":
    #     pass
    elif name in ["resnet18", "resnet50", "resnet101", "micranet", "convnext-tiny", "convnext-small", "convnext-base"]:
        model, cfg = get_base_model(name, in_channels=3 if (weights is not None and "imagenet" in weights.lower()) else 1)
        state_dict = get_weights(name, weights)
        # This is could lead to errors if the model is not exactly the same as the one used for pretraining
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict, strict=False)
        print(f"--- Loaded model {name} with weights {weights} ---")
        return model, cfg
    else:
        raise NotImplementedError(f"Model {name} not implemented yet.")
    return model

def get_classifier_v2(name: str, weights: str, task: str, path: str = None, dataset: str = None, **kwargs):
    if name in ['vit-small', 'vit-base']:
        model, cfg = get_base_model(name, **kwargs)
        state_dict = get_weights(name, weights)
        model.load_state_dict(state_dict, strict=False)
        return model
    elif name in ["resnet18", "resnet50", "resnet101", "micranet", "convnext-tiny", "convnext-small", "convnext-base", "mae", "mae-small"]:
        backbone, cfg = get_base_model(name, **kwargs)
        model = LinearProbe(
            backbone=backbone,
            name=name,
            num_classes=4,
            num_blocks=kwargs['blocks'] # no need to be specific about num_blocks for already trained classifiers
        )
        if path is not None:
            if "imagenet" in weights.lower():
                checkpoint = torch.load(f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/{name.split('-')[0]}_ImageNet/{dataset}/{task}_{path}_model.pth")
            elif "ctc" in weights.lower():
                checkpoint = torch.load(f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/{name.split('-')[0]}_CTC/{dataset}/{task}_{path}_model.pth")
            elif "sted" in weights.lower():
                checkpoint = torch.load(f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/{name.split('-')[0]}_STED/{dataset}/{task}_{path}_model.pth")
            else:
                raise NotImplementedError
            model.load_state_dict(checkpoint['model_state_dict'])
            return model, cfg
        else:
            state_dict = get_weights(name, weights)
            model.load_state_dict(state_dict, strict=False) # Loads the linear probe by default

    else:
        raise NotImplementedError(f"Model {name} not implemented as a classifier yet.")



def get_classifier(name: str, pretraining: str, task:str, path: str = None, dataset: str = None, **kwargs):
    if name == "vit-small":
        print(f"--- Loading ViT-S/16 trained from scratch on {dataset}---")
        model = vit_small_patch16_224(in_chans=1)
        checkpoint = torch.load(f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/MAE_fully-supervised/{dataset}/vit-small_from-scratch_model.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    elif name == "MAE":
        if pretraining == "ImageNet":
            print("--- Loading ImageNet ViT fine-tuned ---")
            vit = vit_small_patch16_224(in_chans=3, pretrained=True)
            backbone = LightlyMAE(vit=vit, in_channels=3, mask_ratio=0.0)
            model = LinearProbe(
                backbone=backbone,
                name="MAE",
                num_classes=4,
                num_blocks=0,
                global_pool='avg'
            )
            if path is not None:
                checkpoint = torch.load(f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/mae_ImageNet/{dataset}/{task}_{path}_model.pth")
            else:
                checkpoint = torch.load(f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/mae_ImageNet/{dataset}/{task}_model.pth")
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
                num_blocks=0,
                global_pool="avg"
            )
            if path is not None:
                checkpoint = torch.load(f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/mae_CTC/{dataset}/{task}_{path}_model.pth")
            else:
                checkpoint = torch.load(f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/mae_CTC/{dataset}/{task}_model.pth")
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
                num_blocks=0,
                global_pool="avg"
            )
            if path is not None:
                checkpoint = torch.load(f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/mae_STED/{dataset}/{task}_{path}_model.pth")
            else:
                checkpoint = torch.load(f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/mae_STED/{dataset}/{task}_model.pth")
            model.load_state_dict(checkpoint['model_state_dict'])
            return model
        else:
            raise NotImplementedError(f"Pretraining {pretraining} not supported.")
    else: 
        raise not NotImplementedError(f"Model {name} not implemented yet.")
