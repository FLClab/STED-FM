from timm.models.vision_transformer import VisionTransformer
import torch

def build_mae_lightning_64_p8(cfg, pretrained=False):
    return VisionTransformer(
        img_size = cfg.img_size,
        patch_size = cfg.patch_size,
        in_chans= cfg.in_chans,
        embed_dim = cfg.embed_dim,
        depth = cfg.depth, # nb of transformer block
        num_heads = cfg.num_heads, # attention heads
        mlp_ratio = 4.0,
        num_classes = 0,
        global_pool = "" # returns all token 
    )

