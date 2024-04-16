
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

from decoders.unet import UNet

import sys 
sys.path.insert(0, "..")

from model_builder import get_base_model

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

    backbone, cfg = get_base_model(args.backbone)
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
