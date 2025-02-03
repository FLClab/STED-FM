
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
import matplotlib.pyplot as plt
from dataclasses import dataclass
from lightly import loss
from lightly import transforms
from lightly.data import LightlyDataset
from lightly.models.modules import heads
from tqdm import tqdm
from collections import defaultdict
from collections.abc import Mapping
from multiprocessing import Manager
from torch.utils.data import Sampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
from torchinfo import summary
from lightly.utils.scheduler import CosineWarmupScheduler

from decoders import get_decoder
from datasets import get_dataset
from eval import evaluate_segmentation

import sys 
sys.path.insert(0, "..")

from model_builder import get_base_model, get_pretrained_model_v2
from utils import update_cfg, save_cfg
from configuration import Configuration
from DEFAULTS import BASE_PATH

def validation_step(model: torch.nn.Module, valid_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, epoch: int, device: torch.device, writer: SummaryWriter = None):
    is_training = model.training

    model.eval()
    
    statLossTest = []
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
    
    if is_training:
        model.train()
    return statLossTest  

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

class RandomNumberOfSamplesSampler(Sampler):
    def __init__(self, cfg: Configuration, data_source: torch.utils.data.Dataset, num_samples_per_class: int, seed: int = None):
        self.cfg = cfg
        self.data_source = data_source
        self.num_samples_per_class = num_samples_per_class
        self.rng = numpy.random.default_rng(seed)

        self.indices = self.__get_indices()
        
    def __get_indices(self):
        indices_per_class = defaultdict(list)
        for i, (image, mask) in enumerate(self.data_source):
            sum_per_class = torch.sum(mask, dim=(1, 2))
            for c, s in enumerate(sum_per_class):
                if s > self.cfg.dataset_cfg.min_annotated_ratio * mask.size(-1) * mask.size(-2):
                    indices_per_class[c].append(i)
        print("Number of samples per class: ", {k: len(v) for k, v in indices_per_class.items()})
        indices = []
        for key, values in indices_per_class.items():
            if len(values) < self.num_samples_per_class:
                print("Warning: Not enough samples for class", key)
                print("Number of samples available: ", len(values))
                print("Number of samples requested: ", self.num_samples_per_class)
                indices.extend(values)
            else:
                indices.extend(self.rng.choice(values, size=self.num_samples_per_class, replace=False))
        return indices
            
    def __iter__(self):
        return iter(self.rng.permutation(self.indices))

    def __len__(self):
        return len(self.indices)

class SegmentationConfiguration(Configuration):
    
    freeze_backbone: bool = True
    num_epochs: int = 300
    learning_rate: float = 1e-4

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")     
    parser.add_argument("--restore-from", type=str, default=None,
                    help="Model from which to restore from") 
    parser.add_argument("--save-folder", type=str, default=f"{BASE_PATH}/segmentation-baselines",
                    help="Model from which to restore from")     
    parser.add_argument("--dataset", required=True, type=str,
                    help="Name of the dataset to use")             
    parser.add_argument("--backbone", type=str, default="resnet18",
                        help="Backbone model to load")
    parser.add_argument("--backbone-weights", type=str, default=None,
                        help="Backbone model to load")    
    parser.add_argument("--use-tensorboard", action="store_true",
                        help="Logging using tensorboard")
    parser.add_argument("--label-percentage", type=float, default=None,
                        help="Percentage of labels to use")
    parser.add_argument("--num-per-class", type=int, default=None,
                        help="Number of samples to use")
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
    update_cfg(cfg, args.opts)

    if args.restore_from:
        # Loads checkpoint
        checkpoint = torch.load(args.restore_from)
        OUTPUT_FOLDER = os.path.dirname(args.restore_from)

        # Loads previous configuration and updates it
        cfg = Configuration.from_json(os.path.join(os.path.dirname(args.restore_from), "config.json"))

        # Updates configuration with additional options; performs inplace
        if args.opts:
            print("Warning: Additional options are updated in the original configuration file. The original configuration file will be overwritten.")
        update_cfg(cfg, args.opts)
        args = cfg.args
        
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
        if args.label_percentage is not None and args.label_percentage < 1.0:
            model_name += f"-{int(args.label_percentage * 100)}%-labels"
        elif args.num_per_class is not None:
            model_name += f"-{args.num_per_class}-samples"

        model_name += f"-{args.seed}"

        OUTPUT_FOLDER = os.path.join(args.save_folder, args.backbone, args.dataset, model_name)

    if args.dry_run and not args.restore_from:
        OUTPUT_FOLDER = os.path.join(args.save_folder, "debug")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(OUTPUT_FOLDER, "logs"))
    # Save and print configuration
    cfg.save(os.path.join(OUTPUT_FOLDER, "config.json"))
    print(cfg)

    # Build the UNet model.
    model = get_decoder(backbone, cfg).to(DEVICE)
    
    ckpt = checkpoint.get("model", None)
    if not ckpt is None:
        print("Restoring model...")
        model.load_state_dict(ckpt)
    model = model.to(DEVICE)

    # Loads stats from checkpoint
    stats = checkpoint.get("stats", None)
    if not stats is None:
        min_valid_loss = numpy.min(stats["testMean"])
        start_epoch = len(stats["trainMean"])
    else:
        stats = defaultdict(list)
        min_valid_loss = numpy.inf
        start_epoch = 0

    # Prints a summary of the model
    # summary(model, input_size=(cfg.in_channels, 224, 224))

    # Sampler definition
    sampler = None
    if args.label_percentage is not None and args.label_percentage < 1.0:
        # Ensures reporoducibility
        rng = numpy.random.default_rng(seed=args.seed)
        indices = list(range(len(training_dataset)))
        rng.shuffle(indices)
        split = int(numpy.floor(args.label_percentage * len(training_dataset)))
        train_indices, _ = indices[:split], indices[split:]
        sampler = SubsetRandomSampler(train_indices)
    elif args.num_per_class is not None:
        sampler = RandomNumberOfSamplesSampler(cfg, training_dataset, args.num_per_class, seed=args.seed)
    
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
        batch_size=cfg.batch_size,  # A large batch size helps with the learning.
        shuffle=sampler is None,  # Shuffling is important!
        num_workers=4,
        sampler=sampler, drop_last=False
    )
    valid_loader = torch.utils.data.DataLoader(
        validation_dataset,  # Pass the dataset to the dataloader.
        batch_size=cfg.batch_size,  # A large batch size helps with the learning.
        shuffle=True,  # Shuffling is important!
        num_workers=4
    )

    # Defines the training budget
    num_epochs = cfg.num_epochs
    if args.label_percentage is not None and args.label_percentage < 1.0:
        budget = len(training_dataset) * num_epochs
        num_epochs = int(budget / (args.label_percentage * len(training_dataset)))
        cfg.num_epochs = num_epochs
        print(f"Training budget is updated: {cfg.num_epochs} epochs")
    elif args.num_per_class is not None:
        budget = len(training_dataset) * num_epochs
        num_epochs = int(budget / (args.num_per_class * len(training_dataset.classes)))
        cfg.num_epochs = num_epochs
        print(f"Training budget is updated: {cfg.num_epochs} epochs")

    optimizer = torch.optim.AdamW(model.parameters(), lr = cfg.learning_rate, weight_decay=1e-2)
    criterion = getattr(torch.nn, cfg.dataset_cfg.criterion)()

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 10, threshold = 0.01, min_lr=1e-5, factor=0.1,)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=cfg.num_epochs, eta_min=1e-5)
    scheduler = CosineWarmupScheduler(
        optimizer=optimizer, warmup_epochs=0.1*cfg.num_epochs, max_epochs=num_epochs,
        start_value=1.0, end_value=0.01
    )

    step = start_epoch * len(train_loader)
    print(start_epoch, step, cfg.num_epochs)

    if cfg.num_epochs > 1000:
        cfg.num_epochs = 1000
    for epoch in range(start_epoch, cfg.num_epochs):

        start = time.time()
        print("[----] Starting epoch {}/{}".format(epoch + 1, cfg.num_epochs))

        # Keep track of the loss of train and test
        statLossTrain = []

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
            if step % int(25 * 32 / cfg.batch_size) == 0:
                # Validation step
                statLossTest = validation_step(model, valid_loader, criterion, epoch, DEVICE, writer)
                for key, func in zip(("testMean", "testMed", "testMin", "testStd"),
                                     (numpy.mean, numpy.median, numpy.min, numpy.std)):
                    stats[key].append(func(statLossTest))
                    if args.use_tensorboard:
                        writer.add_scalar(f"Loss/{key}", stats[key][-1], step)
                stats["testStep"].append(step)
            step += 1

        # Aggregate stats
        for key, func in zip(("trainMean", "trainMed", "trainMin", "trainStd"),
                (numpy.mean, numpy.median, numpy.min, numpy.std)):
            stats[key].append(func(statLossTrain))
            if args.use_tensorboard:
                writer.add_scalar(f"Loss/{key}", stats[key][-1], step)

        # scheduler.step(numpy.min(stats["testMean"]))
        scheduler.step()
        stats["lr"].append(scheduler.get_last_lr())
        if args.use_tensorboard:
            _lr = stats["lr"][-1]
            if isinstance(_lr, list):
                for i in range(len(_lr)):
                    writer.add_scalar(f"Learning-rate/lr-{i}", _lr[i], step)
            else:
                writer.add_scalar(f"Learning-rate/lr", _lr, step)
            writer.add_scalar(f"Epochs/epoch", epoch, step)
        stats["trainStep"].append(step)
 
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

        # # Save every 10 epochs
        # if (epoch + 1) % 100 == 0:
        #     savedata = {
        #         "model" : model.state_dict(),
        #         "optimizer" : optimizer.state_dict(),
        #         "stats" : stats,
        #     }
        #     torch.save(
        #         savedata, 
        #         os.path.join(OUTPUT_FOLDER, f"checkpoint-{epoch + 1}.pt"))
        #     del savedata

    print("----------------------------------------")
    print("Training is over")
    print("Evaluation on the test set")
    print("----------------------------------------")

    del model
    torch.cuda.empty_cache() 

    # Build the UNet model.
    model = get_decoder(backbone, cfg)
    ckpt = torch.load(os.path.join(OUTPUT_FOLDER, "result.pt"))["model"]
    print("Restoring model...")
    model.load_state_dict(ckpt)
    model = model.to(DEVICE)
    model.eval()

    # Build a PyTorch dataloader.
    test_loader = torch.utils.data.DataLoader(
        testing_dataset,  # Pass the dataset to the dataloader.
        batch_size=cfg.batch_size,  # A large batch size helps with the learning.
        shuffle=True,  # Shuffling is important!
        num_workers=0
    )

    scores = evaluate_segmentation(model, test_loader, savefolder=None, device=DEVICE)
    with open(os.path.join(OUTPUT_FOLDER, "segmentation-scores.json"), "w") as file: 
        json.dump(scores, file, indent=4)

    print("----------------------------------------")
    print("Evaluation is over")
    print("----------------------------------------")