
import torch
import torchvision
import numpy
import os
import typing
import random
import dataclasses
import time

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
from datasets import get_dataset

import sys 
sys.path.insert(0, "..")

from model_builder import get_base_model, get_pretrained_model

@dataclass
class SegmentationConfiguration:
    
    batch_size: int = 32
    freeze_backbone: bool = False
    num_epochs: int = 500
    learning_rate: float = 1e-4

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")     
    parser.add_argument("--restore-from", type=str, default="",
                    help="Model from which to restore from") 
    parser.add_argument("--save-folder", type=str, default="./data/SSL/segmentation-baselines",
                    help="Model from which to restore from")     
    parser.add_argument("--dataset", required=True, type=str,
                    help="Model from which to restore from")             
    parser.add_argument("--backbone", type=str, default="resnet18",
                        help="Backbone model to load")
    parser.add_argument("--backbone-weights", type=str, default=None,
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

    # Loads backbone model
    if args.backbone_weights:
        backbone, cfg = get_pretrained_model(args.backbone, weights=args.backbone_weights)
    else:
        backbone, cfg = get_base_model(args.backbone)

    # Loads dataset and dataset-specific configuration
    cache_manager = Manager()
    cache_system = cache_manager.dict()
    training_dataset, validation_dataset, testing_dataset, dataset_cfg = get_dataset(name=args.dataset, cache_system=cache_system)
    for key, value in dataset_cfg.__dict__.items():
        setattr(cfg, key, value)

    # Loads segmentation configuration
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
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(OUTPUT_FOLDER, "logs"))

    # Build the UNet model.
    model = UNet(backbone, cfg)
    ckpt = checkpoint.get("model", None)
    if not ckpt is None:
        print("Restoring model...")
        model.load_state_dict(ckpt)
    model = model.to(DEVICE)

    # Loads stats from checkpoint
    stats = checkpoint.get("stats", None)
    if not stats is None:
        min_valid_loss = numpy.min(stats["testMean"])
    else:
        stats = defaultdict(list)
        min_valid_loss = numpy.inf

    # Prints a summary of the model
    summary(model, input_size=(1, 224, 224))

    # Build a PyTorch dataloader.
    train_loader = torch.utils.data.DataLoader(
        training_dataset,  # Pass the dataset to the dataloader.
        batch_size=cfg.batch_size,  # A large batch size helps with the learning.
        shuffle=True,  # Shuffling is important!
        num_workers=4
    )
    valid_loader = torch.utils.data.DataLoader(
        validation_dataset,  # Pass the dataset to the dataloader.
        batch_size=cfg.batch_size,  # A large batch size helps with the learning.
        shuffle=True,  # Shuffling is important!
        num_workers=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr = cfg.learning_rate)
    criterion = torch.nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 10, threshold = 0.01, min_lr=1e-5, factor=0.1,)
    for epoch in range(cfg.num_epochs):

        start = time.time()
        print("[----] Starting epoch {}/{}".format(epoch + 1, cfg.num_epochs))

        # Keep track of the loss of train and test
        statLossTrain, statLossTest = [], []

        # Puts the model in training mode
        model.train()
        for i, (X, y) in enumerate(tqdm(train_loader, desc="[----] ")):

            # Reshape
            if isinstance(X, (list, tuple)):
                X = [_X.unsqueeze(0) if _X.dim() == 2 else _X for _X in X]
            else:
                if X.dim() == 3:
                    X = X.unsqueeze(1)

            # Send to gpu
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            # Prediction and loss computation
            pred = model.forward(X)
            loss = criterion(pred, y)

            # Keeping track of statistics
            statLossTrain.append(loss.item())

            # Back-propagation and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # To avoid memory leak
            torch.cuda.empty_cache()
            del X, y, pred, loss

        # Puts the model in evaluation mode
        model.eval()
        for i, (X, y) in enumerate(tqdm(valid_loader, desc="[----] ")):

            # Reshape
            if isinstance(X, (list, tuple)):
                X = [_X.unsqueeze(0) if _X.dim() == 2 else _X for _X in X]
            else:
                if X.dim() == 3:
                    X = X.unsqueeze(1)

            # Send to gpu
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            # Prediction and loss computation
            pred = model.forward(X)
            loss = criterion(pred, y)

            # Keeping track of statistics
            statLossTest.append(loss.item())

            # To avoid memory leak
            torch.cuda.empty_cache()
            del X, y, pred, loss

        # Aggregate stats
        for key, func in zip(("trainMean", "trainMed", "trainMin", "trainStd"),
                (numpy.mean, numpy.median, numpy.min, numpy.std)):
            stats[key].append(func(statLossTrain))
            if args.use_tensorboard:
                writer.add_scalar(f"Loss/{key}", stats[key][-1], epoch)

        for key, func in zip(("testMean", "testMed", "testMin", "testStd"),
                (numpy.mean, numpy.median, numpy.min, numpy.std)):
            stats[key].append(func(statLossTest))
            if args.use_tensorboard:
                writer.add_scalar(f"Loss/{key}", stats[key][-1], epoch)

        scheduler.step(numpy.min(stats["testMean"]))
        stats["lr"].append(numpy.array(scheduler.get_last_lr()))
        if args.use_tensorboard:
            writer.add_scalar(f"Learning-rate/lr", stats["lr"][-1].item(), epoch)

        # Save if best model so far
        if min_valid_loss > stats["testMean"][-1]:
            min_valid_loss = stats["testMean"][-1]
            savedata = {
                "model" : model,
                "optimizer" : optimizer,
                "stats" : stats,
            }
            torch.save(
                savedata, 
                os.path.join(OUTPUT_FOLDER, "result.pt"))
            
            del savedata

        # Save every 10 epochs
        if epoch % 10 == 0:
            savedata = {
                "model" : model,
                "optimizer" : optimizer,
                "stats" : stats,
            }
            torch.save(
                savedata, 
                os.path.join(OUTPUT_FOLDER, f"checkpoint-{epoch}.pt"))
            del savedata
