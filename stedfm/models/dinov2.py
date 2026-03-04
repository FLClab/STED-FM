import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
import math

from timm.models.vision_transformer import (
    vit_small_patch16_224,
    vit_tiny_patch16_224,
    vit_base_patch16_224,
    vit_large_patch16_224,
)
from lightly.models.modules.heads import DINOv2ProjectionHead
from lightly.loss import DINOLoss, IBOTPatchLoss, KoLeoLoss
from lightly.models.utils import update_momentum, deactivate_requires_grad, random_block_mask
from lightly.utils.scheduler import cosine_schedule
from lightning.pytorch.core import LightningModule

from stedfm.DEFAULTS import BASE_PATH
from stedfm.configuration import Configuration


class DINOv2Weights:
    DINOV2_TINY_IMAGENET1K_V1 = None
    DINOV2_SMALL_IMAGENET1K_V1 = None
    DINOV2_BASE_IMAGENET1K_V1 = None
    DINOV2_LARGE_IMAGENET1K_V1 = None

    DINOV2_TINY_STED = os.path.join(BASE_PATH, "baselines", "dinov2-tiny_STED", "pl_checkpoint-999.pth")
    DINOV2_SMALL_STED = os.path.join(BASE_PATH, "baselines", "dinov2-small_STED", "pl_checkpoint-999.pth")
    DINOV2_BASE_STED = os.path.join(BASE_PATH, "baselines", "dinov2-base_STED", "pl_checkpoint-999.pth")
    DINOV2_LARGE_STED = os.path.join(BASE_PATH, "baselines", "dinov2-large_STED", "pl_checkpoint-999.pth")

    DINOV2_TINY_JUMP = os.path.join(BASE_PATH, "baselines", "dinov2-tiny_JUMP", "pl_checkpoint-999.pth")
    DINOV2_SMALL_JUMP = os.path.join(BASE_PATH, "baselines", "dinov2-small_JUMP", "pl_checkpoint-999.pth")
    DINOV2_BASE_JUMP = os.path.join(BASE_PATH, "baselines", "dinov2-base_JUMP", "pl_checkpoint-999.pth")
    DINOV2_LARGE_JUMP = os.path.join(BASE_PATH, "baselines", "dinov2-large_JUMP", "pl_checkpoint-999.pth")

    DINOV2_TINY_HPA = os.path.join(BASE_PATH, "baselines", "dinov2-tiny_HPA", "pl_checkpoint-999.pth")
    DINOV2_SMALL_HPA = os.path.join(BASE_PATH, "baselines", "dinov2-small_HPA", "pl_checkpoint-999.pth")
    DINOV2_BASE_HPA = os.path.join(BASE_PATH, "baselines", "dinov2-base_HPA", "pl_checkpoint-999.pth")
    DINOV2_LARGE_HPA = os.path.join(BASE_PATH, "baselines", "dinov2-large_HPA", "pl_checkpoint-999.pth")


class DINOv2Configuration(Configuration):

    backbone: str = "vit-small"
    backbone_weights: str = None
    batch_size: int = 64
    dim: int = 384
    in_channels: int = 1
    pretrained: bool = False
    freeze_backbone: bool = False
    n_local_views: int = 6
    global_crop_size: int = 224
    local_crop_size: int = 96


class ViTBackbone(nn.Module):
    """Thin wrapper around a timm ViT to expose a .vit attribute and .encode() method.
    This ensures compatibility with the existing MAE pipeline (LinearProbe, decoders, etc.)
    where backbone.vit is expected.
    """
    def __init__(self, vit):
        super().__init__()
        self.vit = vit
        self.mask_token = nn.Parameter(torch.zeros(1, 1, vit.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    @property
    def sequence_length(self):
        return self.vit.patch_embed.num_patches

    def _interpolate_pos_embed(self, num_patches, embed_dim):
        """Interpolate positional embeddings for different input sizes."""
        pos_embed = self.vit.pos_embed  # (1, 1+N_train, dim)
        cls_pos = pos_embed[:, :1, :]  # CLS token pos embed
        patch_pos = pos_embed[:, 1:, :]  # patch pos embeds

        N_train = patch_pos.shape[1]
        if num_patches == N_train:
            return pos_embed

        # Reshape to 2D grid, interpolate, flatten back
        H_train = W_train = int(math.sqrt(N_train))
        H_new = W_new = int(math.sqrt(num_patches))
        patch_pos = patch_pos.reshape(1, H_train, W_train, embed_dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(H_new, W_new), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def encode(self, images, mask=None):
        """Forward pass through the ViT, optionally masking patch tokens.

        :param images: (B, C, H, W) input images
        :param mask: Optional bool tensor (B, num_patches). True = mask this patch.
        :returns: (B, 1 + num_patches, embed_dim) encoded tokens (CLS + patches)
        """
        # Disable strict size check for local crops
        old_strict = getattr(self.vit.patch_embed, 'strict_img_size', True)
        self.vit.patch_embed.strict_img_size = False
        x = self.vit.patch_embed(images)
        self.vit.patch_embed.strict_img_size = old_strict

        num_patches = x.shape[1]
        if mask is not None:
            x[mask] = self.mask_token.squeeze(0).squeeze(0)
        cls_tokens = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Interpolate positional embeddings if needed (local crops)
        pos_embed = self._interpolate_pos_embed(num_patches, x.shape[-1])
        x = x + pos_embed

        x = self.vit.pos_drop(x)
        if hasattr(self.vit, 'patch_drop') and self.vit.patch_drop is not None:
            x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x


def get_backbone(name: str, **kwargs) -> torch.nn.Module:
    cfg = DINOv2Configuration()
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    cfg.pretrained = cfg.in_channels == 3

    if name == "dinov2-lightning-tiny":
        cfg.dim = 192
        cfg.batch_size = 64
        cfg.backbone = "vit-tiny"
        vit = vit_tiny_patch16_224(in_chans=cfg.in_channels, pretrained=cfg.pretrained)
        backbone = DINOv2(vit=vit, in_channels=cfg.in_channels, n_local_views=cfg.n_local_views)

    elif name == "dinov2-lightning-small":
        cfg.dim = 384
        cfg.batch_size = 64
        cfg.backbone = "vit-small"
        vit = vit_small_patch16_224(in_chans=cfg.in_channels, pretrained=cfg.pretrained)
        backbone = DINOv2(vit=vit, in_channels=cfg.in_channels, n_local_views=cfg.n_local_views)

    elif name == "dinov2-lightning-base":
        cfg.dim = 768
        cfg.batch_size = 32
        cfg.backbone = "vit-base"
        vit = vit_base_patch16_224(in_chans=cfg.in_channels, pretrained=cfg.pretrained)
        backbone = DINOv2(vit=vit, in_channels=cfg.in_channels, n_local_views=cfg.n_local_views)

    elif name == "dinov2-lightning-large":
        cfg.dim = 1024
        cfg.batch_size = 16
        cfg.backbone = "vit-large"
        vit = vit_large_patch16_224(in_chans=cfg.in_channels, pretrained=cfg.pretrained)
        backbone = DINOv2(vit=vit, in_channels=cfg.in_channels, n_local_views=cfg.n_local_views)

    else:
        raise NotImplementedError(f"`{name}` not implemented")
    return backbone, cfg


class DINOv2(LightningModule):
    def __init__(
        self,
        vit,
        in_channels: int = 1,
        n_local_views: int = 6,
        projection_hidden_dim: int = 2048,
        projection_bottleneck_dim: int = 256,
        projection_output_dim: int = 65536,
        warmup_teacher_temp_epochs: int = 30,
        teacher_temp: float = 0.07,
        student_temp: float = 0.1,
        koleo_weight: float = 0.1,
    ) -> None:
        super().__init__()

        self.n_local_views = n_local_views
        self.koleo_weight = koleo_weight
        embed_dim = vit.embed_dim

        # Student and teacher backbones
        self.student_backbone = ViTBackbone(vit)
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        deactivate_requires_grad(self.teacher_backbone)

        # Alias for downstream compatibility (LinearProbe accesses backbone.backbone.vit)
        self.backbone = self.student_backbone

        # DINO / iBOT projection heads (shared weights by default)
        head_kwargs = dict(
            input_dim=embed_dim,
            hidden_dim=projection_hidden_dim,
            bottleneck_dim=projection_bottleneck_dim,
            output_dim=projection_output_dim,
        )
        self.student_head = DINOv2ProjectionHead(**head_kwargs)
        self.teacher_head = DINOv2ProjectionHead(**head_kwargs)
        # Initialize teacher with student weights
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        deactivate_requires_grad(self.teacher_head)

        # Losses
        self.dino_loss = DINOLoss(
            output_dim=projection_output_dim,
            warmup_teacher_temp=0.04,
            teacher_temp=teacher_temp,
            warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
            student_temp=student_temp,
        )
        self.ibot_loss = IBOTPatchLoss(
            output_dim=projection_output_dim,
            teacher_temp=teacher_temp,
            student_temp=student_temp,
        )
        self.koleo_loss = KoLeoLoss()

    def forward_encoder(self, x, idx_keep=None):
        """Compatibility with MAE interface for downstream feature extraction."""
        return self.student_backbone.encode(x)

    def training_step(self, batch, batch_idx):
        # batch is a list of views: [global_0, global_1, local_0, ..., local_N]
        views = batch
        n_global = 2
        global_views = views[:n_global]
        local_views = views[n_global:]

        # Momentum schedule for EMA
        momentum = cosine_schedule(
            step=self.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=0.992,
            end_value=1.0,
        )
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)

        # --- Teacher forward (global views only, no masking, no grad) ---
        teacher_global = torch.cat(global_views, dim=0)
        with torch.no_grad():
            teacher_out = self.teacher_backbone.encode(teacher_global)
            teacher_cls = teacher_out[:, 0]  # (2B, dim)
            teacher_patch = teacher_out[:, 1:]  # (2B, N, dim)
            teacher_cls_proj = self.teacher_head(teacher_cls)  # (2B, proj_dim)
            teacher_patch_proj = self.teacher_head(
                teacher_patch.reshape(-1, teacher_patch.shape[-1])
            ).reshape(teacher_patch.shape[0], teacher_patch.shape[1], -1)  # (2B, N, proj_dim)

        # --- Block mask for iBOT (applied to student global views) ---
        B = global_views[0].shape[0]
        num_patches = self.student_backbone.sequence_length
        H = W = int(num_patches ** 0.5)
        block_mask = random_block_mask(
            size=(2 * B, H, W),
            device=global_views[0].device,
        )  # (2B, H, W) bool
        patch_mask = block_mask.flatten(start_dim=1)  # (2B, N)

        # --- Student forward (global views with masking) ---
        student_global_out = self.student_backbone.encode(teacher_global, mask=patch_mask)
        student_global_cls = student_global_out[:, 0]
        student_global_patch = student_global_out[:, 1:]
        student_global_cls_proj = self.student_head(student_global_cls)

        # Masked patch projections for iBOT
        student_masked_patch_proj = self.student_head(
            student_global_patch[patch_mask]
        )
        teacher_masked_patch_proj = teacher_patch_proj.reshape(-1, teacher_patch_proj.shape[-1])[
            patch_mask.reshape(-1)
        ]

        # --- Student forward (local views, no masking) ---
        student_local_cls_projs = []
        for local_view in local_views:
            local_out = self.student_backbone.encode(local_view)
            local_cls = local_out[:, 0]
            student_local_cls_projs.append(self.student_head(local_cls))

        # --- Compute losses ---
        # DINO loss: student CLS (all views) vs teacher CLS (global views)
        teacher_cls_list = teacher_cls_proj.chunk(2)  # [teacher_g0, teacher_g1]
        student_cls_list = list(student_global_cls_proj.chunk(2)) + student_local_cls_projs

        dino_loss = self.dino_loss(
            teacher_out=teacher_cls_list,
            student_out=student_cls_list,
            epoch=self.current_epoch,
        )

        # iBOT loss: masked patch predictions (mask must be (B, H, W) for IBOTPatchLoss)
        ibot_loss = self.ibot_loss(
            teacher_out=teacher_masked_patch_proj,
            student_out=student_masked_patch_proj,
            mask=block_mask,
        )

        # KoLeo regularizer on student global CLS tokens
        koleo_loss = self.koleo_weight * sum(
            self.koleo_loss(t) for t in student_global_cls.chunk(2)
        )

        total_loss = dino_loss + ibot_loss + koleo_loss

        self.log("Loss/total", total_loss, on_epoch=True, sync_dist=True)
        self.log("Loss/dino", dino_loss, on_epoch=True, sync_dist=True)
        self.log("Loss/ibot", ibot_loss, on_epoch=True, sync_dist=True)
        self.log("Loss/koleo", koleo_loss, on_epoch=True, sync_dist=True)
        self.log("Momentum", momentum, on_epoch=True, sync_dist=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=1.5e-4, weight_decay=0.04, betas=(0.9, 0.95)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        return [optimizer], [scheduler]
