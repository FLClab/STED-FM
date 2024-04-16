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