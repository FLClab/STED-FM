
import numpy
import torch
import os
import lightly.models.utils
from torch import nn
from timm.models.vision_transformer import vit_small_patch16_224, vit_tiny_patch16_224, vit_base_patch16_224, vit_large_patch16_224, VisionTransformer
from timm.layers.patch_embed import PatchEmbed
from timm.layers.pos_embed import resample_abs_pos_embed
import lightly.models.utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from lightning.pytorch.core import LightningModule
import torchvision
from torchmetrics.image import StructuralSimilarityIndexMeasure

from torch.utils.tensorboard import SummaryWriter

from dataclasses import dataclass
from stedfm.DEFAULTS import BASE_PATH
from stedfm.configuration import Configuration
from stedfm.utils import update_cfg
from stedfm.modules.loss import FourierLoss, kl_loss
from tiffwrapper import make_composite

# from instanseg.utils.models.ChannelInvariantNet import ChannelInvariantNet

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

    MAE_MCMS_SMALL_STED = os.path.join(BASE_PATH, "baselines", "mae-mcms-small_STED", "image-scale-finetuning", "current_model.pth")
    MAE_MCMS_SMALL_STED = os.path.join(BASE_PATH, "baselines", "mae-mcms-small_STED", "current_model.pth")

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
    opts = kwargs.pop("opts", [])
    update_cfg(cfg, opts)

    for key, value in kwargs.items():
        print(key, value)
        setattr(cfg, key, value)
    cfg.pretrained = cfg.in_channels == 3

    if name == "mae-lightning-tiny":
        cfg.dim = 192
        cfg.batch_size = 256
        cfg.backbone = "vit-tiny"
        vit = vit_tiny_patch16_224(in_chans=cfg.in_channels, pretrained=cfg.pretrained) 
        backbone = MAE(vit=vit, in_channels=cfg.in_channels, mask_ratio=cfg.mask_ratio)

    elif name == "mae-lightning-small" or name =="mae-lightning-224-p16":
        cfg.dim = 384
        cfg.batch_size = 256
        cfg.backbone = "vit-small"
        vit = vit_small_patch16_224(in_chans=cfg.in_channels, pretrained=cfg.pretrained)
        backbone = MAE(vit=vit, in_channels=cfg.in_channels, mask_ratio=cfg.mask_ratio)
    
    elif name == "mae-mcms-lightning-small" or name == "mae-mcms-lightning-224-p16":
        cfg.dim = 384
        cfg.batch_size = 256
        cfg.backbone = "vit-small"
        vit = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=cfg.dim,
            in_chans=1,
            dynamic_img_size=True,
            depth=12,
            num_heads=6,
        )
        backbone = MCMSMAE(vit=vit, in_channels=cfg.in_channels, 
                           mask_ratio=cfg.mask_ratio,
                           pretrained_model_name="mae-lightning-small", 
                           pretrained_weights="MAE_SMALL_STED",
                           freeze_backbone=cfg.freeze_backbone)

    elif name == "mae-lightning-base":
        cfg.dim = 768
        cfg.batch_size = 128
        cfg.backbone = "vit-base"
        vit = vit_base_patch16_224(in_chans=cfg.in_channels, pretrained=cfg.pretrained)
        backbone = MAE(vit=vit, in_channels=cfg.in_channels, mask_ratio=cfg.mask_ratio)

    elif name == 'mae-lightning-large':
        cfg.dim = 1024
        cfg.batch_size = 64
        cfg.backbone = "vit-large"
        vit = vit_large_patch16_224(in_chans=cfg.in_channels, pretrained=cfg.pretrained)
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

class MCMSVisionTransformer(VisionTransformer):
    """
    Multi-Channel and Multi-Scale Vision Transformer for STED-FM images. The architecture is based on the standard Vision Transformer, 
    but with modifications to handle multi-channel and multi-scale inputs.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.channel_invariant_net = ChannelInvariantNet()

    def forward_features(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = self.channel_invariant_net(x)
        return super().forward_features(x, *args, **kwargs)
    
    def forward_intermediates(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = self.channel_invariant_net(x)
        return super().forward_intermediates(x, *args, **kwargs)

class MCMSPatchEmbed(PatchEmbed):
    """
    Patch embedding module for the MCMSVisionTransformer.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.pixel_size_embed = nn.Linear(1, kwargs.get('embed_dim', 384))
        self.pixel_size = None

    @classmethod
    def from_patch_embed(cls, patch_embed: PatchEmbed):
        norm_layer = patch_embed.norm.__class__ if patch_embed.norm is not None else None
        new_patch_embed = cls(
            img_size=patch_embed.img_size,
            patch_size=patch_embed.patch_size,
            in_chans=patch_embed.proj.in_channels,
            embed_dim=patch_embed.proj.out_channels,
            norm_layer=norm_layer,
            flatten=patch_embed.flatten,
            output_fmt=patch_embed.output_fmt,
            bias=patch_embed.proj.bias is not None,
            strict_img_size=patch_embed.strict_img_size,
            dynamic_img_pad=patch_embed.dynamic_img_pad,
        )
        new_patch_embed.load_state_dict(patch_embed.state_dict(), strict=False)
        return new_patch_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = super().forward(x)  # Shape [B, num_patches, embed_dim]
        if self.pixel_size is None:
            return patches

        pixel_size = torch.as_tensor(
            self.pixel_size, device=patches.device, dtype=patches.dtype
        )
        if pixel_size.ndim == 0:
            pixel_size = pixel_size.view(1, 1).expand(patches.shape[0], 1)
        elif pixel_size.ndim == 1:
            pixel_size = pixel_size[:, None]
        elif pixel_size.ndim != 2 or pixel_size.shape[1] != 1:
            raise ValueError("pixel_size must be a scalar or a tensor shaped [B] or [B, 1]")

        pixel_size_embedding = self.pixel_size_embed(pixel_size).view(patches.shape[0], 1, 1, -1)  # Shape [B, 1, 1, embed_dim]
        return patches + pixel_size_embedding

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

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projection layers for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, anchor_tokens, other_channels_list):
        """
        anchor_tokens: Shape [B, 197, embed_dim] (Channel 1)
        other_channels_list: List of Tensors, each [B, 197, embed_dim] (Channels 2, 3, ... N)
        """
        B, N_tokens, C = anchor_tokens.shape
        
        # 1. Concatenate all non-anchor channels along the token (sequence) dimension
        # Resulting shape: [B, 197 * (N-1), embed_dim]
        kv_context = torch.cat(other_channels_list, dim=1) 
        
        # 2. Project to Q, K, V matrices
        Q = self.q_proj(anchor_tokens)  # [B, 197, C]
        K = self.k_proj(kv_context)     # [B, 197 * (N-1), C]
        V = self.v_proj(kv_context)     # [B, 197 * (N-1), C]

        # 3. Reshape for Multi-Head Attention
        Q = Q.reshape(B, N_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 4. Compute Scaled Dot-Product Attention
        # Attention map shape: [B, num_heads, 197, 197 * (N-1)]
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 5. Aggregate Values and project back
        out = (attn @ V).permute(0, 2, 1, 3).reshape(B, N_tokens, C)
        return self.out_proj(out) + anchor_tokens  # Residual connection

class SymmetricCrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.cross_attn = CrossAttentionFusion(embed_dim, num_heads) # Reuse the previous block

    def forward(self, channel_tokens_list):
        """
        channel_tokens_list: List of N tensors, each of shape [B, 197, embed_dim]
        """
        N_channels = len(channel_tokens_list)
        fused_outputs = []

        # Loop through every channel, making each one the "Anchor" exactly once
        for i in range(N_channels):
            query_channel = channel_tokens_list[i]
            
            # Gather all OTHER channels to form the Key/Value context pool
            context_channels = [
                channel_tokens_list[j] for j in range(N_channels) if j != i
            ]
            
            # Compute cross-attention for this specific channel's perspective
            updated_channel_tokens = self.cross_attn(query_channel, context_channels)
            fused_outputs.append(updated_channel_tokens)

        # Merge the symmetrically updated tokens. 
        # Average them together (Keeps the shape at [B, 197, embed_dim])
        # average_fused_tokens = torch.stack(fused_outputs, dim=0).mean(dim=0)
        
        return fused_outputs  # Return the list of updated tokens for each channel (each [B, 197, embed_dim])

class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, scale_factor=2):
        super(ResUp, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_in // 2, kernel_size, 1, kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channel_in // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 1, kernel_size // 2)

        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")

        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.up_nn(x)
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return self.act_fnc(self.bn2(x + skip))

class DecoderToPixelSpace(nn.Module):
    def __init__(self, decoder_embed_dim, embed_dim=64, channels=1):
        super().__init__()

        self.conv_t_up = nn.ConvTranspose2d(decoder_embed_dim, embed_dim * 4, 4, 1)
        self.res_up_block1 = ResUp(embed_dim * 4, embed_dim * 2)
        self.res_up_block2 = ResUp(embed_dim * 2, embed_dim)
        self.conv_out = nn.Conv2d(embed_dim, channels, 3, 1, 1)
        self.act_fnc = nn.ELU()        
    
    def forward(self, x):
        
        # Reshape for ConvTranspose2d: [B, num_tokens, embed_dim] -> [B*num_tokens, embed_dim, 1, 1]
        batch_size, num_tokens, embed_dim = x.shape
        x = x.view(batch_size * num_tokens, embed_dim, 1, 1)  # Shape [B*num_tokens, embed_dim, 1, 1]

        x = self.act_fnc(self.conv_t_up(x))  # 4
        x = self.res_up_block1(x)  # 8
        x = self.res_up_block2(x)  # 16
        x = torch.tanh(self.conv_out(x))

        # Reshape back to [B, num_tokens, num_pixels_per_patch]
        x = x.view(batch_size, num_tokens, -1)
        return x 

class MSMAEDecoderTimm(MAEDecoderTIMM):
    def __init__(
        self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def decode(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder transformer.

        Args:
            input:
                Tensor with shape (batch_size, seq_length, hidden_dim) containing
                the encoded tokens.

        Returns:
            Tensor with shape (batch_size, seq_length, hidden_dim) containing
            the decoded tokens.

        """
        # decoder_pos_embed has shape [1, num_patches + 1, decoder_embed_dim], where the first token is the CLS token.
        # We need to resample the positional embeddings to match the sequence length of the input tokens, which may be 
        # different from the original number of patches due to multi-scale inputs. We also need to ensure that the CLS 
        # token embedding is preserved and added to the resampled positional embeddings.
        new_size = int((input.shape[1] - 1) ** 0.5)  # Assuming input shape is [B, seq_length, embed_dim]
        new_size = (new_size, new_size)
        decoder_pos_embed = resample_abs_pos_embed(self.decoder_pos_embed, new_size)  # Shape [1, new_seq_length, decoder_embed_dim]
        
        output: torch.Tensor = input + decoder_pos_embed
        output = self.decoder_blocks(output)
        output = self.decoder_norm(output)
        return output   

class MSMAEVAEDecoderTimm(MAEDecoderTIMM):
    def __init__(
        self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        decoder_embed_dim = self.decoder_pos_embed.shape[-1]
        self.mu_proj = nn.Linear(decoder_embed_dim, decoder_embed_dim)
        self.logvar_proj = nn.Linear(decoder_embed_dim, decoder_embed_dim)

        self.decoder_to_pixel_space = DecoderToPixelSpace(
            decoder_embed_dim=decoder_embed_dim, 
            embed_dim=64, 
            channels=1)
    
    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def predict(self, input: torch.Tensor) -> torch.Tensor:
        
        mu = self.mu_proj(input)
        logvar = self.logvar_proj(input)
        if self.training:
            x = self.sample(mu, logvar)
        else:
            x = mu  # Use the mean during inference for deterministic output

        x = self.decoder_to_pixel_space(x)

        return x, mu, logvar

    def decode(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder transformer.

        Args:
            input:
                Tensor with shape (batch_size, seq_length, hidden_dim) containing
                the encoded tokens.

        Returns:
            Tensor with shape (batch_size, seq_length, hidden_dim) containing
            the decoded tokens.

        """
        # decoder_pos_embed has shape [1, num_patches + 1, decoder_embed_dim], where the first token is the CLS token.
        # We need to resample the positional embeddings to match the sequence length of the input tokens, which may be 
        # different from the original number of patches due to multi-scale inputs. We also need to ensure that the CLS 
        # token embedding is preserved and added to the resampled positional embeddings.
        new_size = int((input.shape[1] - 1) ** 0.5)  # Assuming input shape is [B, seq_length, embed_dim]
        new_size = (new_size, new_size)
        decoder_pos_embed = resample_abs_pos_embed(self.decoder_pos_embed, new_size)  # Shape [1, new_seq_length, decoder_embed_dim]
        
        output: torch.Tensor = input + decoder_pos_embed
        output = self.decoder_blocks(output)
        output = self.decoder_norm(output)
        return output            

class MCMSMAE(MAE):
    def __init__(self, vit, in_channels, mask_ratio: float = 0.75, 
                 pretrained_model_name: str = "mae-lightning-small", 
                 pretrained_weights: str = "MAE_SMALL_STED",
                 freeze_backbone: bool = False) -> None:

        super().__init__(vit=vit, in_channels=1, mask_ratio=mask_ratio)

        # This needs to be done after the super().__init__ call, as the MAE constructor initializes the backbone and decoder based on the provided vit, 
        # so we need to load the pretrained weights into the vit before it is used to initialize the backbone and decoder
        # vit should be pretrained
        from stedfm.models.loading import get_weights
        state_dict = get_weights(pretrained_model_name, pretrained_weights)
        # Only keep the vit weights from the state dict and load them into the vit backbone
        state_dict = {k.replace("backbone.vit.", ""): v for k, v in state_dict.items() if "backbone.vit" in k}
        vit.load_state_dict(state_dict, strict=True)

        # Ensure the backbone vit is updated
        self.backbone.vit = vit
        self.backbone.vit.patch_embed = MCMSPatchEmbed.from_patch_embed(
            self.backbone.vit.patch_embed
        )

        self.decoder = MSMAEVAEDecoderTimm(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=self.decoder.decoder_pos_embed.shape[-1],
            in_chans=1,
            decoder_depth=1,
            decoder_num_heads=8,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0
        )

        # Freeze the backbone if specified
        if freeze_backbone:
            print("[---] Freezing backbone weights except for pixel size embedding layer [---]")
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.vit.patch_embed.pixel_size_embed.requires_grad_(True)

        # Create the fusion module for multi-channel token fusion (this is done within the backbone after the ViT encoder)
        self.fusion_module = SymmetricCrossAttentionFusion(
            embed_dim=vit.embed_dim, 
            num_heads=6
        )

        # Other loss
        self.mse_criterion = self.criterion
        self.fourier_criterion = FourierLoss()
        self.fourier_lambda = 0.01
        self.ssim_criterion = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.ssim_lambda = 0.01
        self.kl_criterion = kl_loss
        self.kl_lambda = 0.1

    def set_pixel_size(self, pixel_size):
        self.backbone.vit.patch_embed.pixel_size = pixel_size

    def forward_batch(self, images):
        batch_size = images.shape[0]
        channels = images.shape[1]

        # Encode each channel separately
        channel_tokens, channel_idx_keeps, channel_idx_masks = [], [], []
        for chan_idx in range(channels):
            # Here we will set the mask_ratio to a distributed into N channels, i.e. if the original mask ratio is 0.75 
            # and we have 3 channels, then each channel will have a mask ratio of 0.75 ** (1/3) = 0.908, so that the overall 
            # masking across all channels is still 0.75
            idx_keep, idx_mask = lightly.models.utils.random_token_mask(
                size=(batch_size, self.sequence_length),
                mask_ratio=self.mask_ratio ** (1 / channels),  # Distribute the masking across channels
                device=images.device
            )
            x_encoded = self.forward_encoder(x=images[:, chan_idx:chan_idx+1], idx_keep=idx_keep)

            channel_tokens.append(x_encoded)
            channel_idx_keeps.append(idx_keep)
            channel_idx_masks.append(idx_mask)

        # Perform Symmetric Cross-Attention fusion of tokens from different channels (this is done within the backbone)
        if channels > 1:
            fused_channel_tokens = self.fusion_module(channel_tokens)
        else:
            fused_channel_tokens = channel_tokens

        return fused_channel_tokens, channel_idx_keeps, channel_idx_masks

    def decode_batch(self, images, fused_channel_tokens, channel_idx_keeps, channel_idx_masks):
        channels = len(fused_channel_tokens)
        batch_size = fused_channel_tokens[0].shape[0]

        mu, logvar = None, None

        # Decode each channel separately using the fused tokens
        loss = 0
        losses = {"mse" : 0}
        for chan_idx in range(channels):
            x_encoded = fused_channel_tokens[chan_idx]
            idx_keep = channel_idx_keeps[chan_idx]
            idx_mask = channel_idx_masks[chan_idx]

            x_pred = self.forward_decoder(x=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask)
            if isinstance(x_pred, tuple):
                x_pred, mu, logvar = x_pred  # Unpack the tuple if using the VAE decoder

            patches = lightly.models.utils.patchify(images[:, chan_idx:chan_idx+1], self.patch_size)
            target = lightly.models.utils.get_at_index(patches, idx_mask-1)

            mse_channel_loss = self.mse_criterion(x_pred, target)
            losses["mse"] += mse_channel_loss

            # MSSSIM/FourierLoss expects images with shape [B, C, H, W], so we need to unpatchify the predicted patches and the target patches before computing the loss
            x_pred_unpatchified = x_pred.view(batch_size, -1, self.patch_size, self.patch_size)  # Shape [B, num_masked_patches, patch_size, patch_size]
            target_unpatchified = target.view(batch_size, -1, self.patch_size, self.patch_size)  # Shape [B, num_masked_patches, patch_size, patch_size]
            ssim_channel_loss = 1 - self.ssim_criterion(x_pred_unpatchified, target_unpatchified)
            if losses.get("ssim") is None:
                losses["ssim"] = 0
            losses["ssim"] += ssim_channel_loss

            fourier_channel_loss = self.fourier_criterion(x_pred_unpatchified, target_unpatchified)
            if losses.get("fourier") is None:
                losses["fourier"] = 0
            losses["fourier"] += fourier_channel_loss

            if mu is not None and logvar is not None:
                kl_channel_loss = self.kl_criterion(mu, logvar)
                if losses.get("kl") is None:
                    losses["kl"] = 0
                losses["kl"] += kl_channel_loss * self.kl_lambda

            loss += mse_channel_loss
            loss += fourier_channel_loss * self.fourier_lambda  # Scale the Fourier loss to balance it with the MSE loss
            loss += ssim_channel_loss * self.ssim_lambda  # Scale the SSIM loss to balance it with the MSE loss
            loss += kl_channel_loss * self.kl_lambda if mu is not None and logvar is not None else 0
        return loss, losses

    @torch.no_grad()
    def reconstruct(self, images, fused_channel_tokens, channel_idx_keeps, channel_idx_masks):
        channels = len(fused_channel_tokens)
        batch_size = fused_channel_tokens[0].shape[0]

        reconstructed_channels = []
        for chan_idx in range(channels):
            x_encoded = fused_channel_tokens[chan_idx]
            idx_keep = channel_idx_keeps[chan_idx]
            idx_mask = channel_idx_masks[chan_idx]

            x_pred = self.forward_decoder(x=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask)
            if isinstance(x_pred, tuple):
                x_pred, _, _ = x_pred  # Unpack the tuple if using the VAE decoder

            # Unpatchify the predicted patches to get the reconstructed image for this channel
            patches = lightly.models.utils.patchify(images[:, chan_idx:chan_idx+1], self.patch_size)
            patches = lightly.models.utils.set_at_index(patches, idx_mask-1, x_pred.type_as(patches))
            reconstructed_channel = lightly.models.utils.unpatchify(patches, self.patch_size, 1)
            reconstructed_channels.append(reconstructed_channel)

        # Stack the reconstructed channels back together to get the final reconstructed image
        reconstructed_image = torch.cat(reconstructed_channels, dim=1)  # Shape [B, C, H, W]
        return reconstructed_image

    def forward_features(self, x: torch.Tensor, pixel_size: float = None) -> torch.Tensor:

        if isinstance(x, list):
            raise NotImplementedError("Batch inference with list of tensors not implemented yet, the model expects a single tensor of shape [B, C, H, W]")

        batch_size = x.shape[0]
        channels = x.shape[1]

        # Encode each channel separately
        channel_tokens = []
        for chan_idx in range(channels):
            if pixel_size is not None:
                self.set_pixel_size(pixel_size)

            # We need to update the sequence_length since the image size may be different for each item in the batch
            self.sequence_length = x.shape[2] // self.patch_size * x.shape[3] // self.patch_size
            self.sequence_length += self.backbone.vit.num_prefix_tokens  # Account for CLS token

            # Ensures we are calling the ViT encoder without masking during inference, as we want to extract features from the entire image
            x_encoded = self.backbone.vit.forward_features(x[:, chan_idx:chan_idx+1]) # Don't apply masking during inference

            channel_tokens.append(x_encoded)

        # Perform Symmetric Cross-Attention fusion of tokens from different channels (this is done within the backbone)
        if channels > 1:
            fused_channel_tokens = self.fusion_module(channel_tokens)
        else:
            fused_channel_tokens = channel_tokens
        
        features = torch.mean(torch.stack(fused_channel_tokens, dim=0), dim=0) # Average the fused tokens across channels to get a single representation (shape [B, 197, embed_dim])
        return features
    
    def training_step(self, batch, batch_idx):
        images = batch
        metadata = None

        reconstructed_images = None

        if isinstance(images, (list, tuple)):
            # If images is a list, then we encode each item from the list separately
            loss = 0
            losses = {"mse" : 0}
            batch_size = len(images)
            for idx in range(batch_size):
                item = images[idx]
                if isinstance(item, (list, tuple)):
                    item, metadata = item
                
                if metadata is not None:
                    # No pixel size will result in a value of 0, which will be embedded by the pixel size embedding layer and added to the patch embeddings, allowing the model to learn to handle images with missing pixel size information
                    pixel_sizes = [float(meta.get("msr-metadata", {}).get("pixel_size", 0.0)) for meta in metadata]
                    pixel_sizes = torch.as_tensor(pixel_sizes, device=item.device, dtype=item.dtype)
                    self.set_pixel_size(pixel_sizes)

                # Ensure item has shape [B, C, H, W] where B=1
                if item.dim() == 3:
                    item = item.unsqueeze(0)

                # We need to update the sequence_length since the image size may be different for each item in the batch
                self.sequence_length = item.shape[2] // self.patch_size * item.shape[3] // self.patch_size
                self.sequence_length += self.backbone.vit.num_prefix_tokens  # Account for CLS token

                # Encode each channel separately
                fused_channel_tokens, channel_idx_keeps, channel_idx_masks = self.forward_batch(item)
                
                batch_loss, batch_losses = self.decode_batch(item, fused_channel_tokens, channel_idx_keeps, channel_idx_masks)
                for key, value in batch_losses.items():
                    if key not in losses:
                        losses[key] = 0
                    losses[key] += value
                
                loss += batch_loss

                if batch_idx == 0 and idx == 0:
                    reconstructed_images = {
                        "source": item,
                        "reconstructed": self.reconstruct(item, fused_channel_tokens, channel_idx_keeps, channel_idx_masks)}
        else:
            if isinstance(images, (list, tuple)):
                images, metadata = images
            if metadata is not None:
                pixel_sizes = [float(meta.get("msr-metadata", {}).get("pixel_size", 0.0)) for meta in metadata]
                pixel_sizes = torch.as_tensor(pixel_sizes, device=images.device, dtype=images.dtype)
                self.set_pixel_size(pixel_sizes)
            
            # We need to update the sequence_length since the image size may be different for each item in the batch
            self.sequence_length = images.shape[2] // self.patch_size * images.shape[3] // self.patch_size
            self.sequence_length += self.backbone.vit.num_prefix_tokens  # Account for CLS token

            # Encode each channel separately
            fused_channel_tokens, channel_idx_keeps, channel_idx_masks = self.forward_batch(images)
            loss, batch_losses = self.decode_batch(images, fused_channel_tokens, channel_idx_keeps, channel_idx_masks)
            
        self.log("Loss/mean", loss, on_epoch=True, sync_dist=True, batch_size=len(images))
        self.log("Loss/min", loss, on_epoch=True, reduce_fx=torch.min, sync_dist=True, batch_size=len(images))
        self.log("Loss/max", loss, on_epoch=True, reduce_fx=torch.max, sync_dist=True, batch_size=len(images))
        for key, value in losses.items():
            self.log(f"Loss/{key}", value, on_epoch=True, sync_dist=True, batch_size=len(images))
        self.log("Learning Rate", self.trainer.optimizers[0].param_groups[0]["lr"], on_epoch=True, sync_dist=True, batch_size=len(images))

        # Logging images
        writer = self.logger.experiment
        if (batch_idx == 0) and reconstructed_images is not None and isinstance(writer, SummaryWriter):
            source = []
            for i in range(reconstructed_images["source"].shape[0]):
                img = reconstructed_images["source"][i].cpu().numpy()
                source.append(make_composite(img, luts=["gray", "green", "magenta", "cyan"][:img.shape[0]], ranges=[(0, 1)] * img.shape[0]))
            reconstructed = []
            for i in range(reconstructed_images["reconstructed"].shape[0]):
                img = reconstructed_images["reconstructed"][i].cpu().numpy()
                reconstructed.append(make_composite(img, luts=["gray", "green", "magenta", "cyan"][:img.shape[0]], ranges=[(0, 1)] * img.shape[0]))
            source = numpy.array(source).transpose(0, 3, 1, 2)  # Convert to NCHW format for TensorBoard
            reconstructed = numpy.array(reconstructed).transpose(0, 3, 1, 2)  # Convert to NCHW format for TensorBoard
            writer.add_images("Images/source", source, self.current_epoch, dataformats="NCHW")
            writer.add_images("Images/pred", reconstructed, self.current_epoch, dataformats="NCHW")

        return loss

    def load_from_checkpoint(self, checkpoint_path, *args, **kwargs):
        state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        self.load_state_dict(state_dict, strict=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1.5e-4, weight_decay=0.05, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        return [optimizer], [scheduler]    