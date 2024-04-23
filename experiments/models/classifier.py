import torch
from typing import List


class LinearProbe(torch.nn.Module):
    def __init__(
            self,
            backbone: torch.nn.Module,
            name: str,
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
        
        if self.name.lower() == "mae":
            feature_dim = 384
        elif self.name == "resnet18":
            feature_dim = 512
        elif self.name == "resnet50":
            feature_dim = 2048
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
        if self.name in ["MAE", "MAEClassifier", 'mae', 'vit-small']:
            print(f"--- Freezing {blocks} ViT blocks ---")
            self.backbone.backbone.mask_token.requires_grad = False
            self.backbone.backbone.vit.cls_token.requires_grad = False
            self.backbone.backbone.vit.pos_embed.requires_grad = False
            for p in self.backbone.backbone.vit.patch_embed.parameters():
                p.requires_grad = False
            for bidx in blocks:
                for p in self.backbone.backbone.vit.blocks[bidx].parameters():
                    p.requires_grad = False
                    
        elif "resnet" in self.name.lower():
            print(f"--- Freezing {blocks} ResNet layers ---")
            for p in self.backbone.conv1.parameters():
                p.requires_grad = False
            for p in self.backbone.bn1.parameters():
                p.requires_grad = False
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
        if self.name in ["MAE", 'mae', "MAEClassifier"]:
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