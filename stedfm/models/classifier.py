import torch
from typing import List, Union, Optional
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

class ClassificationHead(torch.nn.Module):
    def __init__(
            self,
            in_features: int = 384,
            num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.classfication_head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features=in_features, affine=False, eps=1e-6),
            torch.nn.Linear(in_features=in_features, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classfication_head(x)

class LinearProbe(torch.nn.Module):
    """
    Creates a linear probe on top of a given backbone model.

    :param backbone: The backbone model to use for feature extraction.
    :param name: The name of the backbone model.
    :param cfg: Configuration dictionary containing model parameters.
    :param num_classes: The number of output classes for classification.
    :param global_pool: The type of global pooling to use ('avg', 'token', or 'patch').
    :param num_blocks: The number of blocks to freeze in the backbone model ('all', '0', or an integer).
    """
    def __init__(
        self,
        backbone: torch.nn.Module,
        name: str, 
        cfg: dict,
        num_classes: int = 4, 
        global_pool: str = "avg",
        num_blocks: Literal["all", "0"] = "all",
        **kwargs
    ) -> None:
        super().__init__()

        if "mcms" in name.lower():
            self.backbone = backbone
        elif "mae" in name.lower():
            try:  # ViT case with none-ImageNet weights
                # print("--- ViT case with none-ImageNet weights or from scratch ---")
                self.backbone = backbone.backbone.vit 
            except: # ViT case with ImageNet weights
                # print("--- ViT case with ImageNet weights ---")
                self.backbone = backbone 
        else: # CNN case 
            self.backbone = backbone
        self.name = name 
        self.num_classes = num_classes 
        self.global_pool = global_pool
        self.frozen_blocks = num_blocks 

        if self.frozen_blocks == "all":
            # print(f"--- Freezing every parameter in {name} ---")
            for p in self.backbone.parameters():
                p.requires_grad = False

        elif self.frozen_blocks == "0":
            print(f"--- Not freezing any parameters in {name} ---")
        
        else:
            blk_list = list(range(int(num_blocks)))
            self._freeze_blocks(blk_list)

        self.classification_head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features=cfg.dim, affine=False, eps=1e-6),
            torch.nn.Linear(in_features=cfg.dim, out_features=self.num_classes)
        )

    def train(self, mode: bool = True) -> None:
        if self.frozen_blocks == "all":
            # Linear probe only
            self.backbone.eval()
        else:
            self.backbone.train(mode)
        self.classification_head.train(mode)

    def _freeze_blocks(self, blocks: Union[List, int]) -> None:
        raise NotImplementedError("Partial fine-tuning not yet implemented.") 
    
    def forward_features(self, x: torch.Tensor, return_patches: bool = False, **kwargs) -> torch.Tensor:
        if "mae" in self.name.lower():
            features = self.backbone.forward_features(x, **kwargs)
            if self.global_pool == "token":
                out = features[:, 0, :] # class token 

            elif self.global_pool == "avg":
                out = torch.mean(features[:, 1:, :], dim=1) # Average patch tokens

            elif self.global_pool == "patch":
                out = features[:, 1:, :]
            else:
                raise NotImplementedError(f"Invalid `{self.global_pool}` pooling function.")
        else:
            out = self.backbone.forward(x)  
        if return_patches:
            return out, features 
        else:
            return out    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)

        out = self.classification_head(features)
        return out, features

class MetaLinearProbe(LinearProbe):
    """
    Creates a meta linear probe on top of a given backbone model.
    For single channel models, this meta linear probe simply predicts
    each channel independently and pools the predictions across channels.
    For multichannel models, this meta linear probe simply extracts the 
    features as it is designed to be used in the model.

    :param backbone: The backbone model to use for feature extraction.
    :param name: The name of the backbone model.
    :param cfg: Configuration dictionary containing model parameters.
    :param num_classes: The number of output classes for classification.
    :param global_pool: The type of global pooling to use ('avg', 'token', or 'patch').
    :param num_blocks: The number of blocks to freeze in the backbone model ('all', '0', or an integer).
    :param channel_token_pool: The type of pooling to use for channel tokens ('avg' or 'cat').
    """
    def __init__(
        self,
        backbone: torch.nn.Module,
        name: str, 
        cfg: dict,
        num_classes: int = 4, 
        global_pool: Literal["avg", "token", "patch"] = "avg",
        num_blocks: Literal["all", "0"] = "all",
        channel_token_pool: Literal["avg", "cat"] = "avg",
        **kwargs
    ) -> None:
        super().__init__(
            backbone=backbone, 
            name=name,
            cfg=cfg,
            num_classes=num_classes,
            global_pool=global_pool,
            num_blocks=num_blocks)

        self.channel_token_pool = channel_token_pool
    
    def forward_features(self, x: torch.Tensor, return_patches: bool = False, **kwargs) -> torch.Tensor:
        if "mcms" in self.name.lower():
            # This is a multichannel model, so we simply return the features as is.
            return super().forward_features(x, return_patches=return_patches, **kwargs)
        else:
            # We need to predict each channel independently and pool the predictions across channels.
            batch_size, num_channels, H, W = x.shape
            x = x.view(batch_size * num_channels, 1, H, W) # Treat each channel as a separate image
            out = super().forward_features(x, return_patches=return_patches)
            if return_patches:
                features, patch_features = out 
                features = features.view(batch_size, num_channels, -1)
                patch_features = patch_features.view(batch_size, num_channels, patch_features.shape[1], patch_features.shape[2])
            else:
                features = out.view(batch_size, num_channels, -1)
            
            if self.channel_token_pool == "avg":
                features = features.view(batch_size, num_channels, -1).mean(dim=1) # Average across channels
            elif self.channel_token_pool == "cat":
                features = features.view(batch_size, num_channels, -1).reshape(batch_size, -1) # Concatenate across channels
            else:
                raise NotImplementedError(f"Invalid `{self.channel_token_pool}` pooling function.")
            
            if return_patches:
                return features, patch_features
            return features

class OldLinearProbe(torch.nn.Module):
    def __init__(
            self,
            backbone: torch.nn.Module,
            name: str,
            cfg: dict,
            num_classes: int = 4,
            global_pool: str = 'avg',
            num_blocks: int = 0
            ) -> None:
        super().__init__()
        self.backbone = backbone
        self.name = name
        self.num_classes = num_classes 
        self.global_pool = global_pool
        self.num_blocks = num_blocks
        
        if "mae" in self.name.lower():
            feature_dim = cfg.dim
            print(f"--- Freezing default vit pre-blocks ---")
            self.backbone.backbone.mask_token.requires_grad = False
            self.backbone.backbone.vit.cls_token.requires_grad = False
            self.backbone.backbone.vit.pos_embed.requires_grad = False
            for p in self.backbone.backbone.vit.patch_embed.parameters():
                p.requires_grad = False
        elif self.name == "resnet18":
            feature_dim = 512
            for p in self.backbone.conv1.parameters():
                p.requires_grad = False
            for p in self.backbone.bn1.parameters():
                p.requires_grad = False
        elif self.name == "resnet50":
            feature_dim = 2048
            for p in self.backbone.conv1.parameters():
                p.requires_grad = False
            for p in self.backbone.bn1.parameters():
                p.requires_grad = False
        elif self.name == "micranet":
            pass 
        elif self.name == 'convnext':
            pass
        else:
            raise NotImplementedError(f"Backbone {self.name} not supported.")
        
        if num_blocks == 'all':
            for p in self.backbone.parameters():
                p.requires_grad = False
            print(f"--- Freezing all blocks ---")
        elif num_blocks == "0":
            print("--- Not freezing any layers ---")
        elif num_blocks != "0":
                blocks = list(range(int(num_blocks)))
                self._freeze_blocks(blocks)
        else:
            raise NotImplementedError(f"Invalid number ({num_blocks}) of blocks.")

        self.classification_head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features=feature_dim, affine=False, eps=1e-6),
            torch.nn.Linear(in_features=feature_dim, out_features=num_classes)
        )

    def _freeze_blocks(self, blocks):
        if self.name in ["MAE", "MAEClassifier", 'mae', 'vit-small', 'mae-small', 'mae-base', 'mae-tiny']:
            print(f"--- Freezing {blocks} ViT blocks ---")
            for bidx in blocks:
                for p in self.backbone.backbone.vit.blocks[bidx].parameters():
                    p.requires_grad = False
                    
        elif "resnet" in self.name.lower():
            print(f"--- Freezing {blocks} ResNet layers ---")
            if len(blocks) == 1:
                for p in self.backbone.layer1.parameters():
                    p.requires_grad = False
            if len(blocks) > 1:
                for p in self.backbone.layer2.parameters():
                    p.requires_grad = False
            if len(blocks) > 2:
                for p in self.backbone.layer3.parameters():
                    p.requires_grad = False
            if len(blocks) > 3:
                for p in self.backbone.layer4.parameters():
                    p.requires_grad = False
        
        else: 
            raise NotImplementedError(f"Freezing of {self.name} not supported yet.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if "mae" in self.name.lower():
            features = self.backbone.forward_encoder(x)
            if self.global_pool == "token":
                features = features[:, 0, :] # class token
            elif self.global_pool == "avg":
                features = torch.mean(features[:, 1:, :], dim=1) # Average patch tokens
            else:
                exit(f"{self.global_pool} not implemented yet")
        else:
            features = self.backbone.forward(x)
        out = self.classification_head(features)
        return out


class MAEClassificationHead(torch.nn.Module):
    def __init__(
            self, 
            backbone: torch.nn.Module, 
            feature_dim: int = 384, 
            num_classes: int = 4, 
            freeze: bool = True,
            global_pool: str = 'avg',
            ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.global_pool = global_pool
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.classfication_head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features=384, affine=False, eps=1e-6),
            torch.nn.Linear(in_features=feature_dim, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_encoder(x)
        if self.global_pool == "token":
            features = features[:, 0, :] # class token
        elif self.global_pool == "avg":
            features = torch.mean(features[:, 1:, :], dim=1) # Average patch tokens
        else:
            exit(f"{self.global_pool} not implemented yet")
        out = self.classfication_head(features)
        return out