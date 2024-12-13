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
        freeze_decoder: bool = False
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
                if name in ["decoder_norm", "decoder_pred"]:
                    continue
                else:
                    p.requires_grad = False

        self.decodder_pred = torch.nn.Linear(
            self.mae.decoder_embed_dim, # TODO
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        idx_keep, idx_mask = lightly.models.utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=0.0,
            device=images.device
        )
        print(idx_keep.shape, idx_mask.shape)
        x_encoded = self.mae.forward_encoder(x=x, idx_keep=idx_keep)
        x_decode = self.mae.decoder.embed(x_encoded) 
        x_decoded = self.mae.decoder.decode(x_decode)
        x
            

