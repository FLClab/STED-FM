from models.lightly_mae import LightlyMAE
from models.classifier import MAEClassificationHead
from timm.models.vision_transformer import vit_small_patch16_224
import lightly.models.utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
import torch

def get_pretrained_model(name: str, weights: str = None, path: str = None):
    if name == "MAE":
        if weights == "ImageNet":
            vit = vit_small_patch16_224(in_chans=3, pretrained=True)
            model = LightlyMAE(vit=vit, in_channels=3, mask_ratio=0.0)
        elif weights == "CTC":
            vit = vit_small_patch16_224(in_chans=1)
            model = LightlyMAE(vit=vit, in_channels=1, mask_ratio=0.0)
            checkpoint = torch.load("/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/vit_experiments/Datasets/Cell-Tracking-Challenge/baselines/checkpoint-530.pth")
            model.load_state_dict(checkpoint['model'])
        elif weights == "STED":
            vit = vit_small_patch16_224(in_chans=1)
            model = LightlyMAE(vit=vit, in_channels=1, mask_ratio=0.0)
            checkpoint = torch.load("/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/vit_experiments/Datasets/FLCDataset/baselines/checkpoint-530.pth")
            model.load_state_dict(checkpoint['model'])
        else:
            raise NotImplementedError(f"Weights {weights} not supported yet for model {name}.")
    else:
        raise NotImplementedError(f"Model {name} not implemented yet.")
    return model
