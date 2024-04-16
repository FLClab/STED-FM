
import numpy
import torch
import pickle
import os
import json
import h5py 

from torch import nn

class DoubleConvolver(nn.Module):
    """
    Class for the double convolution in the contracting path. The kernel size is
    set to 3x3 and a padding of 1 is enforced to avoid lost of pixels. The convolution
    is followed by a batch normalization and relu.

    :param in_channels: Number of channels in the input tensor
    :param out_channels: Number of channels produced by the convolution
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConvolver, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Contracter(nn.Module):
    """
    Class for the contraction path. Max pooling of the input tensor is
    followed by the double convolution.

    :param in_channels: Number of channels in the input tensor
    :param out_channels: Number of channels produced by the convolution
    """
    def __init__(self, in_channels, out_channels):
        super(Contracter, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConvolver(in_channels=in_channels, out_channels=out_channels)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Expander(nn.Module):
    """
    Class for the expansion path. Upsampling with a kernel size of 2 and stride 2
    is performed and followed by a double convolution following the concatenation
    of the skipping link information from higher layers.

    :param in_channels: Number of channels in the input tensor
    :param out_channels: Number of channels produced by the convolution
    """
    def __init__(self, in_channels, out_channels):
        super(Expander, self).__init__()
        self.expand = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConvolver(in_channels=in_channels, out_channels=out_channels)

    def center_crop(self, links, target_size):
        _, _, links_height, links_width = links.size()
        diff_x = (links_height - target_size[0]) // 2
        diff_y = (links_width - target_size[1]) // 2
        return links[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        x = self.expand(x)
        crop = self.center_crop(bridge, x.size()[2 : ])
        concat = torch.cat([x, crop], 1)
        x = self.conv(concat)
        return x


class UNet(nn.Module):
    """
    Implements a U-Net model for segmentation. The encoder is a backbone model
    that is used to extract features from the input data. The decoder part of 
    the model follows the seminal U-Net architecture with double convolution
    blocks and skip connections.

    The final image is upsampled to the original size of the input image.
    """    
    def __init__(self, backbone, cfg):
        """
        Initializes the `UNet` model

        :param backbone: A `torch.nn.Module` of the backbone model
        :param cfg: A `dict` of the configuration for the backbone model
        :param num_classes: An `int` of the number of classes to segment
        """
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg

        if self.cfg.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        with torch.no_grad():
            layer_sizes = self.get_layer_sizes()

        self.decoder = torch.nn.Sequential(*[
            Expander(in_channels=layer_sizes[i + 1][0], out_channels=layer_sizes[i][0])
            for i in reversed(range(len(layer_sizes) - 1))
        ])
        self.out_conv = torch.nn.Conv2d(in_channels=layer_sizes[0][0], out_channels=self.cfg.num_classes, kernel_size=1)

    def get_layer_sizes(self) -> list[tuple[int, int, int]]:
        """
        Computes the sizes of the layers in the decoder

        :returns : A `list` of the sizes of the intermediate layers
        """
        sizes = []
        _, out = self.forward_encoder(torch.randn(1, 1, 224, 224))
        for o in out:
            sizes.append(o.shape[1:])
        return sizes

    def forward_encoder(self, x : torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass for the encoder of the `UNet` model. This method is used
        to extract the features from the input data. Specific forward passes
        are implemented for each backbone model.

        :param x: A `torch.tensor` of the input data

        :returns : A `torch.tensor` of the output data
                   A `list` of the intermediate outputs
        """
        func = getattr(self, "_forward_" + self.cfg.backbone)
        return func(x)
    
    def _forward_micranet(self, x : torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass for the `MICRANet` backbone

        :parm x: A `torch.tensor` of the input data

        :returns : A `torch.tensor` of the output data
                   A `list` of the intermediate outputs
        """
        out = []
        x = self.backbone.conv1a(x)
        x = self.backbone.bnorm1a(x)
        x = self.backbone.relu(x)
        x = self.backbone.conv1b(x)
        x = self.backbone.bnorm1b(x)
        x = self.backbone.relu(x)
        out.append(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.conv2a(x)
        x = self.backbone.bnorm2a(x)
        x = self.backbone.relu(x)
        x = self.backbone.conv2b(x)
        x = self.backbone.bnorm2b(x)
        x = self.backbone.relu(x)
        out.append(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.conv3a(x)
        x = self.backbone.bnorm3a(x)
        x = self.backbone.relu(x)
        x = self.backbone.conv3b(x)
        x = self.backbone.bnorm3b(x)
        x = self.backbone.relu(x)
        out.append(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.conv4a(x)
        x = self.backbone.bnorm4a(x)
        x = self.backbone.relu(x)
        x = self.backbone.conv4b(x)
        x = self.backbone.bnorm4b(x)
        x = self.backbone.relu(x)   
        out.append(x)

        return x, out 
    
    def _forward_resnet(self, x : torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass for the `ResNet` backbone

        :param x: A `torch.tensor` of the input data

        :returns : A `torch.tensor` of the output data
                   A `list` of the intermediate outputs
        """
        out = []
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        out.append(x)

        x = self.backbone.layer2(x)
        out.append(x)

        x = self.backbone.layer3(x)
        out.append(x)

        x = self.backbone.layer4(x)
        out.append(x)

        return x, out

    def _forward_convnext(self, x : torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass for the `ConvNext` backbone

        :param x: A `torch.tensor` of the input data

        :returns : A `torch.tensor` of the output data
                   A `list` of the intermediate outputs
        """
        out = []
        for i, layer in enumerate(self.backbone.features):
            x = layer(x)
            if ((i + 1) % 2) == 0:
                out.append(x)
        return x, out

    def forward(self, x : torch.Tensor):
        """
        Implements the forward method of `UNet`.
        The output image is upsampled to the original size of the input image.

        :param x: A `torch.tensor` of the input data

        :returns : A `torch.tensor` of the segmented input data
        """
        size = x.shape[-2:]
        # Forward pass through the encoder
        x, out = self.forward_encoder(x)
        for i, layer in enumerate(self.decoder):
            x = layer(x, out[-i-2])
        x = self.out_conv(x)
        x = torch.nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=True)
        return x
