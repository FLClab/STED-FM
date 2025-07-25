
import os
import torch
import torchvision

from torch import nn
from dataclasses import dataclass

from stedfm.DEFAULTS import BASE_PATH
from stedfm.configuration import Configuration

class MICRANetWeights:
    MICRANET_SSL_HPA = os.path.join(BASE_PATH, "baselines", "micranet_HPA", "checkpoint-999.pt")
    MICRANET_SSL_STED = os.path.join(BASE_PATH, "baselines", "micranet_STED", "result.pt")
    MICRANET_SSL_CTC = os.path.join(BASE_PATH, "baselines", "micranet_CTC", "checkpoint-999.pt")

class MICRANetConfiguration(Configuration):
    
    backbone: str = "micranet"
    backbone_weights: str = None
    batch_size: int = 128
    dim: int = 256
    freeze_backbone: bool = False
    in_channels: int = 1

class MICRANet(nn.Module):
    """
    Class for creating the `MICRANet` architecture

    :param grad: (optional) Wheter the gradient should be calculated
    """
    def __init__(self, grad=False, **kwargs):
        super(MICRANet, self).__init__()
        self.grad = grad

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.conv1a = nn.Conv2d(in_channels=kwargs["in_channels"], out_channels=32, kernel_size=3, padding=1)
        self.bnorm1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bnorm1b = nn.BatchNorm2d(32)

        self.conv2a = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bnorm2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bnorm2b = nn.BatchNorm2d(64)

        self.conv3a = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bnorm3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bnorm3b = nn.BatchNorm2d(128)

        self.conv4a = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bnorm4a = nn.BatchNorm2d(256)
        self.conv4b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bnorm4b = nn.BatchNorm2d(256)

        self.global_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256, kwargs["num_classes"])
        )

        self.grads = {}
        self.outputs = {}

    def forward(self, x):
        """
        Implements the forward method of `MICRANet`

        :param x: A `torch.tensor` of the input data

        :returns : A `torch.tensor` of the classified input data
        """
        x = self.conv1a(x)
        x = self.bnorm1a(x)
        x = self.relu(x)
        if self.grad:
            x.register_hook(self.save_grad("1a"))
            self.outputs["1a"] = x.clone().detach().cpu().data.numpy()
        x = self.conv1b(x)
        x = self.bnorm1b(x)
        x = self.relu(x)
        if self.grad:
            x.register_hook(self.save_grad("1b"))
            self.outputs["1b"] = x.clone().detach().cpu().data.numpy()
        x = self.maxpool(x)

        # 64 x 64
        x = self.conv2a(x)
        x = self.bnorm2a(x)
        x = self.relu(x)
        if self.grad:
            x.register_hook(self.save_grad("2a"))
            self.outputs["2a"] = x.clone().detach().cpu().data.numpy()
        x = self.conv2b(x)
        x = self.bnorm2b(x)
        x = self.relu(x)
        if self.grad:
            x.register_hook(self.save_grad("2b"))
            self.outputs["2b"] = x.clone().detach().cpu().data.numpy()
        x = self.maxpool(x)

        # 32 x 32
        x = self.conv3a(x)
        x = self.bnorm3a(x)
        x = self.relu(x)
        if self.grad:
            x.register_hook(self.save_grad("3a"))
            self.outputs["3a"] = x.clone().detach().cpu().data.numpy()
        x = self.conv3b(x)
        x = self.bnorm3b(x)
        x = self.relu(x)
        if self.grad:
            x.register_hook(self.save_grad("3b"))
            self.outputs["3b"] = x.clone().detach().cpu().data.numpy()
        x = self.maxpool(x)

        # 16 x 16
        x = self.conv4a(x)
        x = self.bnorm4a(x)
        x = self.relu(x)
        if self.grad:
            x.register_hook(self.save_grad("4a"))
            self.outputs["4a"] = x.clone().detach().cpu().data.numpy()
        x = self.conv4b(x)
        x = self.bnorm4b(x)
        x = self.relu(x)
        if self.grad:
            x.register_hook(self.save_grad("4b"))
            self.outputs["4b"] = x.clone().detach().cpu().data.numpy()

        x = self.global_pool(x).squeeze()
        x = self.fc(x)
        return x

    def save_grad(self, name):
        """
        Implements a storing method of the gradients

        :param name: A `str` of the name of the layer
        """
        def hook(grad):
            self.grads[name] = grad.cpu().data.numpy()
        return hook

def get_backbone(name: str, **kwargs) -> torch.nn.Module:
    cfg = MICRANetConfiguration()
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    if name == "micranet":
        backbone = MICRANet(in_channels=cfg.in_channels, num_classes=1)
        # Ignore the classification head as we only want the features.
        backbone.fc = torch.nn.Identity()
    else:
        raise NotImplementedError(f"`{name}` not implemented")
    return backbone, cfg
