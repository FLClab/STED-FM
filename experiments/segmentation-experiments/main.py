
import torch
import torchvision
import numpy
import os
import typing
import random
import dataclasses
import time
import json
import argparse
import uuid

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
    
    freeze_backbone: bool = False
    num_epochs: int = 500
    learning_rate: float = 1e-4

def update_cfg(cfg: dataclass, opts: list[str]) -> dataclass:
    """
    Updates the configuration with additional options inplace

    :param cfg: A `dataclass` of the configuration
    :param opts: A `list` of options to update the configuration
    """
    for i in range(0, len(opts), 2):
        key, value = opts[i], opts[i + 1]
        if len(key.split(".")) > 1:
            key, subkey = key.split(".")
            update_cfg(getattr(cfg, key), [subkey, value])
        else:
            setattr(cfg, key, type(getattr(cfg, key))(value))

def save_cfg(cfg: dataclass, path: str):
    """
    Saves the configuration to a file

    :param cfg: A `dataclass` of the configuration
    :param path: A `str` of the path to save the configuration
    """
    out = {}
    for key, value in cfg.__dict__.items():
        if dataclasses.is_dataclass(value):
            out[key] = save_cfg(value, None)
        elif isinstance(value, argparse.Namespace):
            out[key] = value.__dict__
        else:
            out[key] = value

    # Save to file; if path is None, return the dictionary for recursive calls
    if isinstance(path, str):
        json.dump(out, open(path, "w"), indent=4, sort_keys=True)
    return out

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
                    help="Name of the dataset to use")             
    parser.add_argument("--backbone", type=str, default="resnet18",
                        help="Backbone model to load")
    parser.add_argument("--backbone-weights", type=str, default=None,
                        help="Backbone model to load")    
    parser.add_argument("--use-tensorboard", action="store_true",
                        help="Logging using tensorboard")
    parser.add_argument("--opts", nargs="+", default=[], 
                        help="Additional configuration options")
    parser.add_argument("--dry-run", action="store_true",
                        help="Activates dryrun")        
    args = parser.parse_args()

    # Assert args.opts is a multiple of 2
    if len(args.opts) == 1:
        args.opts = args.opts[0].split(" ")
    assert len(args.opts) % 2 == 0, "opts must be a multiple of 2"

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
    training_dataset, validation_dataset, testing_dataset = get_dataset(name=args.dataset, cfg=cfg, cache_system=cache_system)

    # Updates configuration with additional options; performs inplace
    cfg.args = args
    segmentation_cfg = SegmentationConfiguration()
    for key, value in segmentation_cfg.__dict__.items():
        setattr(cfg, key, value)
    update_cfg(cfg, args.opts)
    print(cfg.__dict__)

    if args.restore_from:
        # Loads checkpoint
        checkpoint = torch.load(args.restore_from)
        OUTPUT_FOLDER = os.path.dirname(args.restore_from)
    else:
        checkpoint = {}
        model_name = ""
        if args.backbone_weights:
            model_name += f"pretrained-"
            print(cfg.freeze_backbone)
            if cfg.freeze_backbone:
                model_name += "frozen-"
            model_name += f"{args.backbone_weights}"
        else:
            model_name += "from-scratch"

        OUTPUT_FOLDER = os.path.join(args.save_folder, args.backbone, args.dataset, model_name)
    if args.dry_run:
        OUTPUT_FOLDER = os.path.join(args.save_folder, "debug")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(OUTPUT_FOLDER, "logs"))

    # Save configuration
    save_cfg(cfg, os.path.join(OUTPUT_FOLDER, "config.json"))

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
        start_epoch = len(stats["testMean"])
    else:
        stats = defaultdict(list)
        min_valid_loss = numpy.inf
        start_epoch = 0

    # Prints a summary of the model
    summary(model, input_size=(cfg.in_channels, 224, 224))

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
    criterion = getattr(torch.nn, cfg.dataset_cfg.criterion)()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 10, threshold = 0.01, min_lr=1e-5, factor=0.1,)
    for epoch in range(start_epoch, cfg.num_epochs):

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
