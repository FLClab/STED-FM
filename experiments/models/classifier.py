import torch
from typing import List, Union

class LinearProbe(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        name: str, 
        cfg: dict,
        num_classes: int = 4, 
        global_pool: str = "avg",
        num_blocks: int = 0,
    ) -> None:
        super().__init__()

        self.backbone = backbone.backbone.vit if "mae" in name.lower() else backbone
        self.name = name 
        self.num_classes = num_classes 
        self.global_pool = global_pool
        self.frozen_blocks = num_blocks 

        if self.frozen_blocks == "all":
            print(f"--- Freezing every parameter in {name} ---")
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

    def _freeze_blocks(self, blocks: Union[List, int]) -> None:
        raise NotImplementedError("Partial fine-tuning not yet implemented.") 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if "mae" in self.name.lower():
            features = self.backbone.forward_features(x)
            if self.global_pool == "token":
                features = features[:, 0, :] # class token 
            elif self.global_pool == "avg":
                features = torch.mean(features[:, 1:, :], dim=1) # Average patch tokens
            else:
                raise NotImplementedError(f"Invalid `{self.global_pool}` pooling function.")
        else:
            features = self.backbone.forward(x)

        out = self.classification_head(features)
        return out, features


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