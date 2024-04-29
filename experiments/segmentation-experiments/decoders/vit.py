import numpy as np  
import torch
from dataclasses import dataclass
from typing import List


class SingleDeconv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0)


    def forward(self, x: torch.Tensor):
        return self.block(x)

class SingleConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.block = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=((kernel_size -1) // 2))

    def forward(self, x: torch.Tensor):
        return self.block(x)

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels,  kernel_size=3):
        super().__init__()
        self.block = torch.nn.Sequential(
            SingleConv(in_channels, out_channels, kernel_size),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(True)
        )

    def forward(self, x: torch.Tensor):
        return self.block(x)

class DeconvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.block = torch.nn.Sequential(
            SingleDeconv(in_channels, out_channels),
            SingleConv(out_channels, out_channels, kernel_size=3),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(True)
        )

    def forward(self, x: torch.Tensor):
        return self.block(x) 

class ViTDecoder(torch.nn.Module):
    def __init__(self,  backbone: torch.nn.Module, cfg: dataclass, extract_layers: List = [3, 6, 9, 12], **kwargs):
        super().__init__()
        self.backbone = backbone
        self.extract_layers = extract_layers
        embed_dim = self.backbone.vit.embed_dim
        self.cfg = cfg
        if self.cfg.freeze_backbone:
            print(f"--- Freezing backbone ---")
            for p in self.backbone.parameters():
                p.requires_grad = False


        self.decoder12_upsampler = SingleDeconv(embed_dim, 512)

        self.decoder9 = torch.nn.Sequential(
            DeconvBlock(embed_dim, 512)
        )
        self.decoder9_upsampler = torch.nn.Sequential(
            ConvBlock(1024, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            SingleDeconv(512, 256)
        )

        self.decoder6 = torch.nn.Sequential(
            DeconvBlock(embed_dim, 512),
            DeconvBlock(512, 256)
        )
        self.decoder6_upsampler = torch.nn.Sequential(
            ConvBlock(512, 256),
            ConvBlock(256, 256),
            SingleDeconv(256, 128)
        )
        
        self.decoder3 = torch.nn.Sequential(
            DeconvBlock(embed_dim, 512),
            DeconvBlock(512, 256),
            DeconvBlock(256, 128)
        )
        self.decoder3_upsampler = torch.nn.Sequential(
            ConvBlock(256, 128),
            ConvBlock(128, 128),
            SingleDeconv(128, 64)
        )

        self.decoder0 = torch.nn.Sequential(
            ConvBlock(self.cfg.in_channels, 32, 3),
            ConvBlock(32, 64, 3)
        )
        self.decoder0_predict = torch.nn.Sequential(
            ConvBlock(128, 64),
            ConvBlock(64, 64),
            SingleConv(64, self.cfg.dataset_cfg.num_classes, 1)
        )

    def forward_encoder(self, x: torch.Tensor):
        x = self.backbone.vit.patch_embed.proj(x)
        B, Hp, Wp = x.shape[0], x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        x = self.backbone.add_prefix_tokens(x)
        x = self.backbone.add_pos_embed(x)
        x = self.backbone.vit.norm_pre(x)
        features = []
        for i, blk in enumerate(self.backbone.vit.blocks):
            x = blk(x)
            if (i+1) in self.extract_layers:
                feat = x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                features.append(feat)
        return feat, features

    def forward(self, x: torch.Tensor):
        size = x.shape[-2:]
        z0 = x
        out, features = self.forward_encoder(x)
        z3, z6, z9, z12 = features

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        pred = self.decoder0_predict(torch.cat([z0, z3], dim=1))
        return pred

def get_decoder(backbone: torch.nn.Module, cfg: dataclass, **kwargs) -> torch.nn.Module:
    """
    Creates a `ViTDecoder` instance

    :param backbone: A `torch.nn.Module` instance
    :param cfg: A `dataclass` instance

    :returns : A `ViTDecoder` instance
    """
    if cfg.backbone in ["vit-small", "mae-small"]:
        extract_layers = [3, 6, 9 ,12]
        return ViTDecoder(backbone.backbone, cfg, extract_layers=extract_layers, **kwargs)
    else:
        raise ValueError(f"Backbone {cfg.backbone} for decoder is not supported")