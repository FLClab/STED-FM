import torch

class LinearProbe(torch.nn.Module):
    def __init__(
            self,
            backbone: torch.nn.Module,
            name: str,
            num_classes: int = 4,
            freeze: bool = True,
            global_pool: str = None,
            ) -> None:
        super().__init__()
        self.backbone = backbone
        self.name = name
        self.num_classes = num_classes 
        self.global_pool = global_pool
        
        if self.name == "MAE":
            feature_dim = 384

        elif self.name == "resnet-18":
            feature_dim = 256
        elif self.name == "resnet-50":
            feature_dim = 512
        elif self.name == "micranet":
            pass 
        elif self.name == 'convnext':
            pass
        else:
            raise NotImplementedError(f"Backbone {self.name} not supported.")
        
        if freeze: # Linear probing, otherwise end-to-end fine-tuning
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.classification_head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features=feature_dim, affine=False, eps=1e-6),
            torch.nn.Linear(in_features=feature_dim, out_features=num_classes)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.name == "MAE":
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