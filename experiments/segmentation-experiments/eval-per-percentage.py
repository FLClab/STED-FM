
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
from collections import defaultdict
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
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score
from matplotlib import pyplot

from decoders import get_decoder
from datasets import get_dataset

import sys 

from stedfm.model_builder import get_pretrained_model_v2
from stedfm.utils import update_cfg, save_cfg, savefig
from stedfm.configuration import Configuration

def comptue_iou(truth: numpy.ndarray, prediction: numpy.ndarray, mask: numpy.ndarray, **kwargs) -> list:
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
        p = p > 0.25

        if (not numpy.any(t) and not numpy.any(p)) or numpy.sum(t) < 0.1 * t.size:
            iou_per_class.append(-1)
            continue 
        if numpy.unique(t).size == 1:
            iou_per_class.append(-1)
            continue        

        intersection = numpy.logical_and(t, p).sum()
        union = numpy.logical_or(t, p).sum()

        iou_per_class.append(intersection / union)
    return iou_per_class

def compute_aupr(truth: numpy.ndarray, prediction: numpy.ndarray, mask: numpy.ndarray, **kwargs) -> list:
    """
    Compute the area under the precision-recall curve between the truth and the prediction

    :param truth: A `numpy.ndarray` of the ground truth
    :param prediction: A `numpy.ndarray` of the prediction
    :param mask: A `numpy.ndarray` of the mask to apply

    :return: A `list` of the AUPR
    """
    aupr_per_class = []
    for t, p in zip(truth, prediction):
        t, p = t[mask].ravel(), p[mask].ravel()

        if not numpy.any(t) or numpy.sum(t) < 0.1 * t.size:
            aupr_per_class.append(-1)
            continue
        if numpy.unique(t).size == 1:
            aupr_per_class.append(-1)
            continue

        precision, recall, _ = precision_recall_curve(t, p)
        
        # From the definition of AUPR, we need to compute the maximum precision for each recall value
        ax = kwargs.get("ax", None)
        if ax:
            ax.plot(recall, precision, color="k", alpha=0.1, rasterized=True)

        aupr_per_class.append(auc(recall, precision))
    return aupr_per_class

def compute_auroc(truth: numpy.ndarray, prediction: numpy.ndarray, mask: numpy.ndarray, **kwargs) -> list:
    """
    Compute the area under the receiver operating characteristic curve between the truth and the prediction

    :param truth: A `numpy.ndarray` of the ground truth
    :param prediction: A `numpy.ndarray` of the prediction
    :param mask: A `numpy.ndarray` of the mask to apply

    :return: A `list` of the AUROC
    """
    auroc_per_class = []
    for t, p in zip(truth, prediction):
        t, p = t[mask].ravel(), p[mask].ravel()
        if not numpy.any(t) or numpy.sum(t) < 0.1 * t.size:
            auroc_per_class.append(-1)
            continue
        if numpy.unique(t).size == 1:
            auroc_per_class.append(-1)
            continue

        auroc_per_class.append(roc_auc_score(t, p))
    return auroc_per_class

def compute_scores(truth: torch.Tensor, prediction: torch.Tensor, **kwargs) -> dict:
    """
    Compute the prediction between the truth and the prediction

    :param truth: A `torch.Tensor` of the ground truth
    :param prediction: A `torch.Tensor` of the prediction
    
    :returns : A `dict` of the computed scores
    """
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

        scores["iou"].append(comptue_iou(truth_, prediction_, mask))
        scores["aupr"].append(compute_aupr(truth_, prediction_, mask, **kwargs))
        scores["auroc"].append(compute_auroc(truth_, prediction_, mask))

    return scores

def evaluate_segmentation(model: torch.nn.Module, loader: torch.utils.data.DataLoader) -> dict:
    """
    Evaluates the segmentation model on the given loader

    :param model: A `torch.nn.Module` of the model to evaluate
    :param loader: A `torch.utils.data.DataLoader` of the loader to evaluate

    :returns : A `dict` of the computed scores
    """
    # fig, ax = pyplot.subplots(figsize=(3,3))
    all_scores = defaultdict(list)
    for i, (X, y) in enumerate(tqdm(loader, desc="[----] ")):

        # Reshape
        if isinstance(X, (list, tuple)):
            X = [_X.unsqueeze(0) if _X.dim() == 2 else _X for _X in X]
        else:
            if X.dim() == 3:
                X = X.unsqueeze(1)

        # Send to gpu
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(X)

        if i == 0:
            X_ = X.cpu().data.numpy()
            y_ = y.cpu().data.numpy()
            pred_ = pred.cpu().data.numpy()

            # import tifffile
            # tifffile.imwrite("./results/input.tif", X_.astype(numpy.float32))
            # tifffile.imwrite("./results/label.tif", y_.astype(numpy.float32))
            # tifffile.imwrite("./results/prediction.tif", pred_.astype(numpy.float32))
        
        scores = compute_scores(y, pred)
        for key, values in scores.items():
            all_scores[key].extend(values)

        del X, y, pred

    # savefig(fig, "./results/pr-curve", save_white=True)

    return all_scores

def plot_scores(scores: dict, metric: str = "aupr", **kwargs):
    """
    Plots the scores from the evaluation of the model

    :param scores: A `dict` of the scores to plot
    :param metric: A `str` of the metric to plot

    :returns : A `tuple` of the figure and axis
    """
    data = []
    for ckpt, performance in scores.items():
        values = numpy.array(performance[metric])
        per_class_data = []
        for i in range(values.shape[1]):
            per_class_values = values[:, i]
            # Remove -1 values as they are not valid
            per_class_values = per_class_values[per_class_values != -1]
            mean = numpy.mean(per_class_values, axis=0)
            per_class_data.append(mean)
        data.append(per_class_data)

    x = numpy.array([int(ckpt) for ckpt in scores.keys()])
    data = numpy.array(data)

    classes = kwargs.get("classes", [str(i) for i in range(data.shape[-1])])

    fig, ax = pyplot.subplots(figsize=(3, 3))
    for i in range(data.shape[-1]):
        ax.plot(x, data[:, i], label=classes[i], marker='o')
    ax.plot(x, numpy.mean(data, axis=-1), label="Mean", marker='o', linestyle='--', color="k")
    ax.set(
        xlabel="Percentage", ylabel=metric,
        ylim=(0, 1)
    )
    ax.legend()

    return fig, ax

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")     
    parser.add_argument("--restore-from", type=str, default="", required=True,
                    help="Folder containing the models from which to restore from") 
    parser.add_argument("--dataset", required=True, type=str,
                    help="Name of the dataset to use")             
    parser.add_argument("--backbone", type=str, default="resnet18",
                        help="Backbone model to load")
    parser.add_argument("--backbone-weights", type=str, default=None,
                        help="Backbone model to load")    
    parser.add_argument("--opts", nargs="+", default=[], 
                        help="Additional configuration options")
    args = parser.parse_args()

    # Assert args.opts is a multiple of 2
    if len(args.opts) == 1:
        args.opts = args.opts[0].split(" ")
    assert len(args.opts) % 2 == 0, "opts must be a multiple of 2"
    # Ensure backbone weights are provided if necessary
    if args.backbone_weights in (None, "null", "None", "none"):
        args.backbone_weights = None

    # Makes sure that args.restore_from is a valid
    if args.restore_from.endswith(os.path.sep):
        args.restore_from = args.restore_from[:-1]    

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    numpy.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Try loading cfg file from restore-from arugments
    config_file = os.path.join(os.path.dirname(args.restore_from), "config.json")
    if os.path.isfile(config_file):
        cfg = Configuration.from_json(config_file)
        args.backbone = cfg["args"]["backbone"]
        args.backbone_weights = cfg["args"]["backbone_weights"]
    else:
        assert args.backbone is not None, "Backbone must be provided"

    backbone, cfg = get_pretrained_model_v2(
        name=args.backbone, 
        weights=args.backbone_weights,
    )
    update_cfg(cfg, args.opts)

    # Loads dataset and dataset-specific configuration
    _, _, testing_dataset = get_dataset(name=args.dataset, cfg=cfg, test_only=True)

    # Build a PyTorch dataloader.
    test_loader = torch.utils.data.DataLoader(
        testing_dataset,  # Pass the dataset to the dataloader.
        batch_size=cfg.batch_size,  # A large batch size helps with the learning.
        shuffle=True,  # Shuffling is important!
        num_workers=0
    )
    
    scores = {}
    for ckpt_idx in [1, 10, 25, 50, 100]:
        print("Evaluating checkpoint: ", ckpt_idx)        
        # Loads checkpoint
        try:
            suffix = f"-{ckpt_idx}%-labels" if ckpt_idx != 100 else ""
            checkpoint = torch.load(os.path.join(args.restore_from + suffix, f"result.pt"))
        except FileNotFoundError:
            print("Checkpoint not found...")
            continue

        # Build the UNet model.
        model = get_decoder(backbone, cfg)
        ckpt = checkpoint.get("model", None)
        if not ckpt is None:
            print("Restoring model...")
            model.load_state_dict(ckpt)
        model = model.to(DEVICE)

        # Puts the model in evaluation mode
        model.eval()
        performance = evaluate_segmentation(model, test_loader)
        scores[ckpt_idx] = performance

        del model

        print("Done evaluating checkpoint: ", ckpt_idx)

    savedir = f"./results/{args.backbone}/{args.dataset}/{os.path.basename(args.restore_from)}"
    os.makedirs(savedir, exist_ok=True)
    
    fig, ax = plot_scores(scores, metric="aupr", classes=testing_dataset.classes)
    savefig(fig, os.path.join(savedir, "aupr-per-percentage"), save_white=True)

    fig, ax = plot_scores(scores, metric="auroc", classes=testing_dataset.classes)
    savefig(fig, os.path.join(savedir, "auroc-per-percentage"), save_white=True)