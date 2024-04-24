import torch 
import numpy as np
import argparse  
from tqdm import tqdm 
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim
from torch.utils.data import DataLoader
from decoders.unet import UNet
from datasets import get_dataset
import sys 
sys.path.insert(0, "../")
from model_builder import get_base_model, get_pretrained_model_v2

def validation_step():
    pass

def train_one_epoch():
    pass

def train():
    pass

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_channels = 3 if "imagenet" in args.weights.lower() else 1 

    backbone, cfg = get_pretrained_model_v2(
        name=args.backbone, 
        weights=args.weights,
        as_classifier=False,
        blocks='all',
        path=None,
        mask_ratio=0.0, 
        pretrained=True if "imagenet" in args.weights.lower() else 1,
        in_channels=n_channels,
        )

    train_dataset, val_dataset, _ = get_dataset(
        name=args.dataset, cfg=cfg)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=6
    )
    valid_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=6,
    )

    model = UNet(backbone, cfg)

    optimizer = torch.optim.Adam(model.parameters, lr=1e-4)
    criterion = torch.nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, patience=10, threshold=0.01, min_lr=1e-5, factor=0.9)

    # TODO: training script


if __name__=="__main__":
    main()