
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
from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from decoders import get_decoder
from datasets import get_dataset

import sys 
sys.path.insert(0, "..")

from model_builder import get_base_model, get_pretrained_model_v2
from utils import update_cfg, save_cfg
from configuration import Configuration

def intensity_scale_(images: torch.Tensor) -> numpy.ndarray:
    """
    Helper function to scale the intensity of the images

    :param images: A `torch.Tensor` of the images to scale

    :returns : A `numpy.ndarray` of the scaled images
    """
    images = images.cpu().data.numpy()
    images = numpy.array([
        (image - image.min()) / (image.max() - image.min()) for image in images
    ])
    return images

class SegmentationConfiguration(Configuration):
    
    freeze_backbone: bool = True
    num_epochs: int = 100
    learning_rate: float = 0.001

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
    parser.add_argument("--label-percentage", type=float, default=1.0,
                        help="Percentage of labels to use")
    parser.add_argument("--opts", nargs="+", default=[], 
                        help="Additional configuration options")
    parser.add_argument("--dry-run", action="store_true",
                        help="Activates dryrun")        
    args = parser.parse_args()

    # Assert args.opts is a multiple of 2
    if len(args.opts) == 1:
        args.opts = args.opts[0].split(" ")
    assert len(args.opts) % 2 == 0, "opts must be a multiple of 2"
    # Ensure backbone weights are provided if necessary
    if args.backbone_weights in (None, "null", "None", "none"):
        args.backbone_weights = None

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    numpy.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Loads backbone model
    if args.backbone_weights:
        backbone, cfg = get_pretrained_model_v2(args.backbone, weights=args.backbone_weights)
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
    # update_cfg(cfg, args.opts)
    # print(cfg.__dict__)
    cfg.backbone_weights = args.backbone_weights
    print(f"Config: {cfg.__dict__}")


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
        if args.label_percentage < 1.0:
            model_name += f"-{int(args.label_percentage * 100)}%-labels"

        OUTPUT_FOLDER = os.path.join(args.save_folder, args.backbone, args.dataset, model_name)
    
    if args.dry_run:
        OUTPUT_FOLDER = os.path.join(args.save_folder, "debug")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(OUTPUT_FOLDER, "logs"))

    # Save configuration
    cfg.save(os.path.join(OUTPUT_FOLDER, "config.json"))

    # Build the UNet model.
    model = get_decoder(backbone, cfg)
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
    # summary(model, input_size=(cfg.in_channels, 224, 224))

    # Sampler definition
    if args.label_percentage < 1.0:
        # Ensures reporoducibility
        rng = numpy.random.default_rng(seed=args.seed)
        indices = list(range(len(training_dataset)))
        rng.shuffle(indices)
        split = int(numpy.floor(args.label_percentage * len(training_dataset)))
        train_indices, _ = indices[:split], indices[split:]
        sampler = SubsetRandomSampler(train_indices)
    else:
        sampler = None
    
    print("----------------------------------------")
    print("Training Dataset")
    print("Dataset size: ", len(training_dataset))
    print("Dataset size (with sampler): ", len(sampler) if sampler else len(training_dataset))
    print("----------------------------------------")
    print("Validation Dataset")
    print("Dataset size: ", len(validation_dataset))
    print(f"Batch size: {cfg.batch_size}")
    print("----------------------------------------")

    # Build a PyTorch dataloader.
    train_loader = torch.utils.data.DataLoader(
        training_dataset,  # Pass the dataset to the dataloader.
        batch_size=32,  # A large batch size helps with the learning.
        shuffle=sampler is None,  # Shuffling is important!
        num_workers=4,
        sampler=sampler, drop_last=False
    )
    valid_loader = torch.utils.data.DataLoader(
        validation_dataset,  # Pass the dataset to the dataloader.
        batch_size=32,  # A large batch size helps with the learning.
        shuffle=True,  # Shuffling is important!
        num_workers=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr = cfg.learning_rate)
    criterion = getattr(torch.nn, cfg.dataset_cfg.criterion)()

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 10, threshold = 0.01, min_lr=1e-5, factor=0.1,)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=cfg.num_epochs, eta_min=1e-5)
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

            if (i == 0) and args.use_tensorboard:
                writer.add_images("Images-train/image", intensity_scale_(X[:16]), epoch, dataformats="NCHW")
                for i in range(cfg.dataset_cfg.num_classes):
                    writer.add_images(f"Images-train/label-{i}", y[:16, i:i+1], epoch, dataformats="NCHW")
                    writer.add_images(f"Images-train/pred-{i}", pred[:16, i:i+1], epoch, dataformats="NCHW")

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

            if (i == 0) and args.use_tensorboard:
                writer.add_images("Images-test/image", intensity_scale_(X[:16]), epoch, dataformats="NCHW")
                for i in range(cfg.dataset_cfg.num_classes):
                    writer.add_images(f"Images-test/label-{i}", y[:16, i:i+1], epoch, dataformats="NCHW")
                    writer.add_images(f"Images-test/pred-{i}", pred[:16, i:i+1], epoch, dataformats="NCHW")            

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

        # scheduler.step(numpy.min(stats["testMean"]))
        scheduler.step()
        stats["lr"].append(numpy.array(scheduler.get_last_lr()))
        if args.use_tensorboard:
            writer.add_scalar(f"Learning-rate/lr", stats["lr"][-1].item(), epoch)

        # Save if best model so far
        if min_valid_loss > stats["testMean"][-1]:
            min_valid_loss = stats["testMean"][-1]
            savedata = {
                "model" : model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "stats" : stats,
            }
            torch.save(
                savedata, 
                os.path.join(OUTPUT_FOLDER, "result.pt"))
            
            del savedata

        # Save every 10 epochs
        if (epoch + 1) % 10 == 0:
            savedata = {
                "model" : model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "stats" : stats,
            }
            torch.save(
                savedata, 
                os.path.join(OUTPUT_FOLDER, f"checkpoint-{epoch + 1}.pt"))
            del savedata
