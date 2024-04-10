import torch

class ClassificationHead(torch.nn.Module):
    def __init__(self, backbone, input_channels: int = 512, output_channels: int = 1, dropout_rate: float = 0.10, freeze: bool = True) -> None:
        super().__init__()
        self.backbone = backbone
        self.input_channels = input_channels 
        self.output_channels = output_channels
        if freeze: 
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.fc1 = torch.nn.Linear(in_features=input_channels, out_features=input_channels // 2)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc2 = torch.nn.Linear(in_features=input_channels//2, out_features=output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        out = self.fc1(features)
        out = self.dropout(out)
        out = self.fc2(out)
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