import numpy as np 
import torch 
from dataclasses import dataclass 
from typing import List 
from lightning.pytorch.core import LightningModule
import lightly.models.utils


class MAEDecoderWrapper(torch.nn.Module):
    def __init__(
        self,
        mae: torch.nn.Module, 
        cfg: dataclass,
        freeze_decoder: bool = True,
        **kwargs
    ) -> None:
        super().__init__()
        self.mae = mae
        self.cfg = cfg
        if self.cfg.freeze_backbone:
            for p in self.mae.backbone.parameters():
                p.requires_grad = False

        if freeze_decoder:
            for p, name in self.mae.decoder.named_parameters():
                # if name in ["decoder_norm", "decoder_pred"]:
                #     continue
                # else:
                p.requires_grad = False

        self.decoder_pred = torch.nn.Linear(
            in_features=self.mae.decoder_embed_dim, 
            out_features=self.mae.patch_size ** 2 * self.mae.decoder.in_chans,
            bias=True 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        idx_keep, idx_mask = lightly.models.utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=0.0,
            device=x.device
        )
        print(idx_keep.shape, idx_mask.shape)
        x_encoded = self.mae.forward_encoder(x=x, idx_keep=idx_keep)
        x_decode = self.mae.decoder.embed(x_encoded) 
        x_decoded = self.mae.decoder.decode(x_decode)
        x_pred = self.decoder_pred(x_decoded)
        return x_pred 

def get_decoder(backbone: torch.nn.Module, cfg: dataclass, **kwargs) -> torch.nn.Module:
    """
    Creates a `MAEDecoderWrapper` instance

    :param backbone: A `torch.nn.Module` instance
    :param cfg: A `dataclass` instance

    :returns : A `MAEDecoderWrapper` instance
    """
    if "mae" in cfg.backbone:
        return MAEDecoderWrapper(
            mae=backbone,
            cfg=cfg, 
            **kwargs
        )
    else:
        raise ValueError(f"Backbone {cfg.backbone} for decoder is not supported")

""" Below:
From https://github.com/lightly-ai/lightly/blob/master/lightly/models/utils.py
"""


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Converts a batch of input images into patches.

    Args:
        images:
            Images tensor with shape (batch_size, channels, height, width)
        patch_size:
            Patch size in pixels. Image width and height must be multiples of
            the patch size.

    Returns:
        Patches tensor with shape (batch_size, num_patches, channels * patch_size ** 2)
        where num_patches = image_width / patch_size * image_height / patch_size.

    """
    # N, C, H, W = (batch_size, channels, height, width)
    N, C, H, W = images.shape
    assert H == W and H % patch_size == 0

    patch_h = patch_w = H // patch_size
    num_patches = patch_h * patch_w

    # Reshape images to form patches
    patches = images.reshape(shape=(N, C, patch_h, patch_size, patch_w, patch_size))

    # Reorder dimensions for patches
    patches = torch.einsum("nchpwq->nhwpqc", patches)

    # Flatten patches
    patches = patches.reshape(shape=(N, num_patches, patch_size**2 * C))

    return patches


def unpatchify(
    patches: torch.Tensor, patch_size: int, channels: int = 3
    ) -> torch.Tensor:
    """
    Reconstructs images from their patches.

     Args:
         patches:
             Patches tensor with shape (batch_size, num_patches, channels * patch_size ** 2).
         patch_size:
             The patch size in pixels used to create the patches.
         channels:
             The number of channels the image must have

     Returns:
         Reconstructed images tensor with shape (batch_size, channels, height, width).
    """
    N, C = patches.shape[0], channels
    patch_h = patch_w = int(patches.shape[1] ** 0.5)
    assert patch_h * patch_w == patches.shape[1]

    images = patches.reshape(shape=(N, patch_h, patch_w, patch_size, patch_size, C))
    images = torch.einsum("nhwpqc->nchpwq", images)
    images = images.reshape(shape=(N, C, patch_h * patch_size, patch_h * patch_size))
    return images
            

