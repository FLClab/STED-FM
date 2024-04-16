
import torch
import torchvision
import numpy
import os
import typing
import random
import dataclasses

from dataclasses import dataclass
from lightly import loss
from lightly import transforms
from lightly.data import LightlyDataset
from lightly.models.modules import heads
from tqdm import tqdm
from collections import defaultdict
from collections.abc import Mapping
from multiprocessing import Manager
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import sys 
sys.path.insert(0, "../simclr-experiments")

import backbones

from dataset import TarFLCDataset
from modules.transforms import SimCLRTransform
from backbones import get_backbone

from decoders.unet import Expander

class UNet(torch.nn.Module):
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

@dataclass
class SegmentationConfiguration:
    
    freeze_backbone: bool = False
    num_classes: int = 1

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")     
    parser.add_argument("--restore-from", type=str, default="",
                    help="Model from which to restore from") 
    parser.add_argument("--save-folder", type=str, default="./data/SSL/baselines",
                    help="Model from which to restore from")     
    parser.add_argument("--dataset-path", type=str, default="./data/FLCDataset/20240214-dataset.tar",
                    help="Model from which to restore from")         
    parser.add_argument("--backbone", type=str, default="resnet18",
                        help="Backbone model to load")
    parser.add_argument("--use-tensorboard", action="store_true",
                        help="Logging using tensorboard")    
    parser.add_argument("--dry-run", action="store_true",
                        help="Activates dryrun")        
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    numpy.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    backbone, cfg = get_backbone(args.backbone)
    segmentation_cfg = SegmentationConfiguration()
    for key, value in segmentation_cfg.__dict__.items():
        setattr(cfg, key, value)

    if args.restore_from:
        checkpoint = torch.load(args.restore_from)
        OUTPUT_FOLDER = os.path.dirname(args.restore_from)
    else:
        checkpoint = {}
        OUTPUT_FOLDER = os.path.join(args.save_folder, args.backbone)
    if args.dry_run:
        OUTPUT_FOLDER = os.path.join(args.save_folder, "debug")
    
    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(OUTPUT_FOLDER, "logs"))

    # Build the UNet model.
    model = UNet(backbone, cfg)
    ckpt = checkpoint.get("model", None)
    if not ckpt is None:
        print("Restoring model...")
        model.load_state_dict(ckpt)
    model = model.to(DEVICE)

    summary(model, input_size=(1, 224, 224))

    # Create a dataset from your image folder.
    dataset = ...

    # Build a PyTorch dataloader.
    dataloader = torch.utils.data.DataLoader(
        dataset,  # Pass the dataset to the dataloader.
        batch_size=cfg.batch_size,  # A large batch size helps with the learning.
        shuffle=True,  # Shuffling is important!
        num_workers=4
    )
