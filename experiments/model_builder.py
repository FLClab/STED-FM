
import os

from models.lightly_mae import LightlyMAE
from models.classifier import LinearProbe
from timm.models.vision_transformer import vit_small_patch16_224
import lightly.models.utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
import torch
import torchvision

from models import get_model
from models.loading import get_weights
from DEFAULTS import BASE_PATH

def get_base_model(name: str, **kwargs):
    model, cfg = get_model(name, **kwargs)
    return model, cfg


def get_pretrained_model_v2(name: str, weights: str = None, as_classifier: bool = False, path: str = None, **kwargs):
    if name in ["resnet18", "resnet50", "resnet101", "micranet", "convnext-tiny", "convnext-small", "convnext-base", "vit-small", "mae-lightning-tiny", "mae-lightning-small", 'mae-lightning-base', 'mae-lightning-large']:
        if "in_channels" not in kwargs:
            kwargs["in_channels"] = 3 if (weights is not None and "imagenet" in weights.lower()) else 1
        backbone, cfg = get_base_model(name, **kwargs)
        state_dict = get_weights(name, weights)
        # This is could lead to errors if the model is not exactly the same as the one used for pretraining
        if weights is None:
            print("--- Loaded model from scratch ---")
        elif state_dict is not None:
            backbone.load_state_dict(state_dict, strict=True)
            print(f"--- Loaded model {name} with weights {weights} ---")
        elif "imagenet" in weights.lower():
            # No state dict to load b/c ImageNet weights were loaded inside the get_weights function
            print(f"--- Loaded model {name} with ImageNet weights ---")
     
        if as_classifier:
            model = LinearProbe(
                backbone=backbone,
                name=name,
                num_classes=kwargs['num_classes'],
                cfg=cfg,
                num_blocks=kwargs['blocks'],
                global_pool=kwargs.get("global_pool", "avg")
            )
            print(f"--- Added linear probe to {kwargs['blocks']} frozen blocks ---")
            return model, cfg
        else:
            return backbone, cfg
    else:
        raise NotImplementedError(f"Model {name} not implemented yet.")
    
def get_classifier_v3(name: str, dataset: str, pretraining: str, **kwargs):
    if "supervised" in name.lower() and "mae" in name.lower():
        modelname = name.replace("supervised-", "")
        backbone, cfg = get_base_model(modelname, **kwargs)
        backbone = backbone.backbone.vit
        modelname = modelname.replace("-lightning", "")
        path = f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/supervised/{modelname}/{dataset}/supervised.pth"
        checkpoint = torch.load(path)
        backbone.load_state_dict(checkpoint["model_state_dict"])
        return backbone, cfg

    elif name.lower() in ["resnet18", "resnet50", "resnet101", "micranet", "convnext-tiny", "convnext-small", "convnext-base", "vit-small", "mae", "mae-tiny", "mae-small", "mae-base", "mae-lightning-tiny", "mae-lightning-small", 'mae-lightning-base', 'mae-lightning-large']:
        if "in_channels" not in kwargs:
            kwargs["in_channels"] = 3 if (pretraining is not None and "imagenet" in pretraining.lower()) else 1        
        backbone, cfg = get_base_model(name, **kwargs)

        # Defines the probe
        probe = kwargs.get("probe", "linear-probe")

        model = LinearProbe(
            backbone=backbone,
            name=name,
            num_classes=kwargs["num_classes"],
            cfg=cfg,
            num_blocks=kwargs['blocks'],
            global_pool=kwargs.get("global_pool", "avg")
        )

        # Loading the weights
        modelname = name.replace("-lightning", "")
        path = os.path.join(BASE_PATH, "baselines", f"{modelname}_{pretraining}", dataset, f"{probe}")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        
        print(f"--- Loaded linear probe weights onto {name} ---")

        return model, cfg
    else:
        raise NotImplementedError(f"Cannot yet add a linear probe to `{name}`.")

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
                checkpoint = torch.load(f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/{name}_ImageNet/{dataset}/{task}_{path}_model.pth")
            elif "ctc" in weights.lower():
                checkpoint = torch.load(f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/{name}_CTC/{dataset}/{task}_{path}_model.pth")
            elif "sted" in weights.lower():
                checkpoint = torch.load(f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/baselines/{name}_STED/{dataset}/{task}_{path}_model.pth")
            else:
                raise NotImplementedError
            model.load_state_dict(checkpoint['model_state_dict'])
            return model, cfg
        else:
            state_dict = get_weights(name, weights)
            model.load_state_dict(state_dict, strict=True) # Loads the linear probe by default

    else:
        raise NotImplementedError(f"Model {name} not implemented as a classifier yet.")

