
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
from collections import defaultdict
from lightly.utils.scheduler import CosineWarmupScheduler
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

def compute_iou(truth: numpy.ndarray, prediction: numpy.ndarray, mask: numpy.ndarray, threshold: float = 0.25) -> list:
    """
    Compute the intersection over union between the truth and the prediction

    :param truth: A `numpy.ndarray` of the ground truth
    :param prediction: A `numpy.ndarray` of the prediction
    :param mask: A `numpy.ndarray` of the mask to apply

    :return: A `list` of the IoU
    """
    iou_per_class = []
    for t, p in zip(truth, prediction):
        t, p = t[mask].ravel(), p[mask].ravel()
        p = p > threshold

        if not numpy.any(t) and not numpy.any(p):
            iou_per_class.append(-1)
            continue 
        if numpy.unique(t).size == 1:
            iou_per_class.append(-1)
            continue        

        intersection = numpy.logical_and(t, p).sum()
        union = numpy.logical_or(t, p).sum()

        iou_per_class.append(intersection / union)
    return iou_per_class

def compute_scores(truth, prediction, threshold):
    truth = truth.cpu().data.numpy()
    prediction = prediction.cpu().data.numpy()
    
    # Case of foreground stored in truth
    if truth.shape[1] != prediction.shape[1]:
        truth, foreground = truth[:, :-1], truth[:, -1]
    else:
        foreground = numpy.ones((len(truth), *truth.shape[-2:]))

    scores = defaultdict(list)
    for truth_, prediction_, mask in zip(truth, prediction, foreground):
        # Convert to binary mask
        mask = mask > 0

        scores["iou"].append(compute_iou(truth_, prediction_, mask, threshold))
    return scores

@dataclass
class SegmentationConfiguration:
    
    freeze_backbone: bool = False
    num_epochs: int = 200
    learning_rate: float = 1e-4 #0.001

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")     
    parser.add_argument("--restore-from", type=str, default="",
                    help="Model from which to restore from") 
    parser.add_argument("--save-folder", type=str, default="/home/koles2/scratch/ssl_project/segmentation_baselines_test", #"./data/SSL/segmentation-baselines/fewshot",
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
    summary(model, input_size=(cfg.in_channels, 224, 224))

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
    print("----------------------------------------")

    # Build a PyTorch dataloader.
    train_loader = torch.utils.data.DataLoader(
        training_dataset,  # Pass the dataset to the dataloader.
        batch_size=64, #cfg.batch_size,  # A large batch size helps with the learning.
        shuffle=sampler is None,  # Shuffling is important!
        num_workers=4,
        sampler=sampler, drop_last=False
    )
    valid_loader = torch.utils.data.DataLoader(
        validation_dataset,  # Pass the dataset to the dataloader.
        batch_size=64, #cfg.batch_size,  # A large batch size helps with the learning.
        shuffle=True,  # Shuffling is important!
        num_workers=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr = cfg.learning_rate)

    # optimizer = torch.optim.SGD(model.parameters(), lr = cfg.learning_rate)
    
    # optimizer = torch.optim.AdamW(model.parameters(), lr = cfg.learning_rate)
    criterion = getattr(torch.nn, cfg.dataset_cfg.criterion)()

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 10, threshold = 0.01, min_lr=1e-5, factor=0.1,)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=cfg.num_epochs, eta_min=1e-5)

    scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup_epochs=10, max_epochs=cfg.num_epochs, start_value=1.0, end_value=0.001)
    
    thresholds = numpy.arange(0.1, 1.0, 0.1)
    all_scores = defaultdict(list)
    print("Training")
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
        print("end training")

        # Puts the model in evaluation mode
        print("Evaluation")
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

            # # Threshold validation
            # for threshold in thresholds:
            #     scores = compute_scores(y, pred, threshold=threshold)
            #     for key, values in scores.items():
            #         values = numpy.array(values)
            #     values = values[values != -1]
            #     # all_scores[threshold] = numpy.mean(values)
            #     all_scores[threshold].append(numpy.mean(values))

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
                #"validation_threshold": best_threshold
            }
            torch.save(
                savedata, 
                os.path.join(OUTPUT_FOLDER, "result.pt"))
            
            del savedata

        # Save every 10 epochs
        if (epoch + 1) % 50 == 0:
            savedata = {
                "model" : model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "stats" : stats,
                #"validation_threshold": best_threshold
            }
            torch.save(
                savedata, 
                os.path.join(OUTPUT_FOLDER, f"checkpoint-{epoch + 1}.pt"))
            del savedata

    # mean_scores = {threshold: numpy.mean(values) for threshold, values in all_scores.items()}
    # print("mean IoU scores for each threshold:")
    # print(mean_scores)
    # optimal_threshold = max(mean_scores, key=mean_scores.get)
    # print(f"optimal Threshold: {optimal_threshold}")
    # print("end validation")