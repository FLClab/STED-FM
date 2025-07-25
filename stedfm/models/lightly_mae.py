import torch
import os
import lightly.models.utils
from timm.models.vision_transformer import vit_small_patch16_224, vit_tiny_patch16_224, vit_base_patch16_224, vit_large_patch16_224, VisionTransformer
import lightly.models.utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from lightning.pytorch.core import LightningModule

from dataclasses import dataclass
from stedfm.DEFAULTS import BASE_PATH
from stedfm.configuration import Configuration

class MAEWeights:
    # IMAGENET pretraining in timm refers to a model pretrained on ImageNet21K and finetuned on ImageNet1K
    # For consistency across the library, we will refer to the model as IMAGENET1K_V1
    MAE_TINY_IMAGENET1K_V1 = None
    MAE_SMALL_IMAGENET1K_V1 = None
    MAE_BASE_IMAGENET1K_V1 = None
    MAE_LARGE_IMAGENET1K_V1 = None

    MAE_64_P8 = os.path.join(BASE_PATH, "baselines", "mae-small_64-p8", "checkpoint-999.pth")
    MAE_224_P16 = os.path.join(BASE_PATH, "baselines", "mae-small_224-p16", "checkpoint-999.pth")

    MAE_TINY_STED = os.path.join(BASE_PATH, "baselines", "mae-tiny_STED", "pl_checkpoint-999.pth")
    # MAE_SMALL_STED = os.path.join(BASE_PATH, "baselines", "mae-small_STED", "pl_checkpoint-999.pth")
    MAE_SMALL_STED = "https://s3.valeria.science/flclab-foundation-models/models/mae-small-sted.zip"
    MAE_BASE_STED = os.path.join(BASE_PATH, "baselines", "mae-base_STED", "pl_checkpoint-999.pth")
    MAE_LARGE_STED = os.path.join(BASE_PATH, "baselines", "mae-large_STED", "pl_checkpoint-999.pth")

    MAE_TINY_JUMP = os.path.join(BASE_PATH, "baselines", "mae-tiny_JUMP", "pl_checkpoint-999.pth")
    MAE_SMALL_JUMP = os.path.join(BASE_PATH, "baselines", "mae-small_JUMP", "checkpoint-999.pth")
    MAE_BASE_JUMP = os.path.join(BASE_PATH, "baselines", "mae-base_JUMP", "pl_checkpoint-999.pth")
    MAE_LARGE_JUMP = os.path.join(BASE_PATH, "baselines", "mae-large_JUMP", "pl_checkpoint-999.pth")

    MAE_TINY_HPA = os.path.join(BASE_PATH, "baselines", "mae-tiny_HPA", "pl_checkpoint-999.pth")
    MAE_SMALL_HPA = os.path.join(BASE_PATH, "baselines", "mae-small_HPA", "pl_checkpoint-999.pth")
    MAE_BASE_HPA = os.path.join(BASE_PATH, "baselines", "mae-base_HPA", "pl_checkpoint-999.pth")
    MAE_LARGE_HPA = os.path.join(BASE_PATH, "baselines", "mae-large_HPA", "pl_checkpoint-999.pth")

    MAE_SMALL_SIM = os.path.join(BASE_PATH, "baselines", "mae-small_SIM", "checkpoint-999.pth")

    MAE_SMALL_HYBRID = os.path.join(BASE_PATH, "baselines", "mae-small_Hybrid", "checkpoint-999.pth")

class MAEConfiguration(Configuration):

    backbone: str = "vit-small"
    backbone_weights: str = None
    batch_size: int = 256
    dim: int = 384
    in_channels: int = 1
    mask_ratio: float = 0.75
    pretrained: bool = False
    freeze_backbone: bool = False

def get_backbone(name: str, **kwargs) -> torch.nn.Module:
    """
    Note that for lightning modules we modify batch size and number of nodes so as to obtain an effective batch
    size of 1024 for all models
    """
    cfg = MAEConfiguration()
    for key, value in kwargs.items():
        print(key, value)
        setattr(cfg, key, value)
    cfg.pretrained = cfg.in_channels == 3

    if name == "mae-lightning-tiny":
        cfg.dim = 192
        cfg.batch_size = 256
        cfg.backbone = "vit-tiny"
        vit = vit_tiny_patch16_224(in_chans=cfg.in_channels, pretrained=cfg.pretrained)
        if cfg.pretrained:
            backbone = vit
        else:
            backbone = MAE(vit=vit, in_channels=cfg.in_channels, mask_ratio=cfg.mask_ratio)

    elif name == "mae-lightning-small" or name =="mae-lightning-224-p16":
        cfg.dim = 384
        cfg.batch_size = 256
        cfg.backbone = "vit-small"
        vit = vit_small_patch16_224(in_chans=cfg.in_channels, pretrained=cfg.pretrained)
        if cfg.pretrained:
            backbone = vit
        else:
            backbone = MAE(vit=vit, in_channels=cfg.in_channels, mask_ratio=cfg.mask_ratio)

    elif name == "mae-lightning-base":
        cfg.dim = 768
        cfg.batch_size = 128
        cfg.backbone = "vit-base"
        vit = vit_base_patch16_224(in_chans=cfg.in_channels, pretrained=cfg.pretrained)
        if cfg.pretrained:
            backbone = vit
        else:
            backbone = MAE(vit=vit, in_channels=cfg.in_channels, mask_ratio=cfg.mask_ratio)

    elif name == 'mae-lightning-large':
        cfg.dim = 1024
        cfg.batch_size = 64
        cfg.backbone = "vit-large"
        vit = vit_large_patch16_224(in_chans=cfg.in_channels, pretrained=cfg.pretrained)
        if cfg.pretrained:
            backbone = vit
        else:
            backbone = MAE(vit=vit, in_channels=cfg.in_channels, mask_ratio=cfg.mask_ratio)

    elif name == "mae-lightning-64-p8":
        print("[---] Using vit-64-p8 backbone [---]")
        cfg.dim = 128 
        cfg.batch_size = 512
        cfg.backbone = "vit-64-p8"
        vit = VisionTransformer(
            img_size=64,
            patch_size=8,
            in_chans=cfg.in_channels,
            num_classes=4,
            embed_dim=cfg.dim,
            depth=8,
            num_heads=4,
        )
        backbone = MAE(vit=vit, in_channels=cfg.in_channels, mask_ratio=cfg.mask_ratio)

    else:
        raise NotImplementedError(f"`{name}` not implemented")
    return backbone, cfg


class MAE(LightningModule):
    def __init__(self, vit, in_channels, mask_ratio: float = 0.75) -> None:
        super().__init__()
        decoder_dim = 512
        self.mask_ratio = mask_ratio 
        self.patch_size = vit.patch_embed.patch_size[0]
        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length
        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=decoder_dim,
            in_chans=in_channels,
            decoder_depth=1,
            decoder_num_heads=8,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0
        )
        self.criterion = torch.nn.MSELoss()

    def forward_encoder(self, x: torch.Tensor, idx_keep: bool = None):
        return self.backbone.encode(images=x, idx_keep=idx_keep)
    
    def forward_decoder(self, x: torch.Tensor, idx_keep: bool, idx_mask: bool):
        batch_size = x.shape[0]
        x_decode = self.decoder.embed(x)
        x_masked = lightly.models.utils.repeat_token(self.decoder.mask_token, (batch_size, self.sequence_length))
        x_masked = lightly.models.utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))
        x_decoded = self.decoder.decode(x_masked)
        x_pred = lightly.models.utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def training_step(self, batch, batch_idx):
        images = batch 
        batch_size = images.shape[0]
        idx_keep, idx_mask = lightly.models.utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device
        )
        x_encoded = self.forward_encoder(x=images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(x=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask)
        patches = lightly.models.utils.patchify(images, self.patch_size)
        target = lightly.models.utils.get_at_index(patches, idx_mask-1)
        loss = self.criterion(x_pred, target)
        self.log("Loss/mean", loss, on_epoch=True, sync_dist=True)
        self.log("Loss/min", loss, on_epoch=True, reduce_fx=torch.min, sync_dist=True)
        self.log("Loss/max", loss, on_epoch=True, reduce_fx=torch.max, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1.5e-4, weight_decay=0.05, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        return [optimizer], [scheduler]


        