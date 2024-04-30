
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
# from torchsummary import summary
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score
from matplotlib import pyplot

from decoders import get_decoder
from datasets import get_dataset

import sys 
sys.path.insert(0, "..")

from model_builder import get_pretrained_model_v2
from utils import update_cfg, save_cfg

def get_save_folder() -> str:
    if "imagenet" in args.backbone_weights.lower():
        return "ImageNet"
    elif "sted" in args.backbone_weights.lower():
        return "STED"
    else:
        return "CTC"

def compute_iou(truth: numpy.ndarray, prediction: numpy.ndarray, mask: numpy.ndarray) -> list:
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

def compute_aupr(truth: numpy.ndarray, prediction: numpy.ndarray, mask: numpy.ndarray) -> list:
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

        if not numpy.any(t):
            aupr_per_class.append(-1)
            continue
        if numpy.unique(t).size == 1:
            aupr_per_class.append(-1)
            continue

        precision, recall, _ = precision_recall_curve(t, p)
        aupr_per_class.append(auc(recall, precision))
    return aupr_per_class

def compute_auroc(truth: numpy.ndarray, prediction: numpy.ndarray, mask: numpy.ndarray) -> list:
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
        if not numpy.any(t):
            auroc_per_class.append(-1)
            continue
        if numpy.unique(t).size == 1:
            auroc_per_class.append(-1)
            continue

        auroc_per_class.append(roc_auc_score(t, p))
    return auroc_per_class

def compute_scores(truth: torch.Tensor, prediction: torch.Tensor) -> dict:
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
        foreground = numpy.ones(truth.shape[-2:])
    
    scores = defaultdict(list)
    for truth_, prediction_, mask in zip(truth, prediction, foreground):
        # Convert to binary mask
        mask = mask > 0

        scores["iou"].append(compute_iou(truth_, prediction_, mask))
        scores["aupr"].append(compute_aupr(truth_, prediction_, mask))
        scores["auroc"].append(compute_auroc(truth_, prediction_, mask))

    return scores

def evaluate_segmentation(model: torch.nn.Module, loader: torch.utils.data.DataLoader) -> dict:
    """
    Evaluates the segmentation model on the given loader

    :param model: A `torch.nn.Module` of the model to evaluate
    :param loader: A `torch.utils.data.DataLoader` of the loader to evaluate

    :returns : A `dict` of the computed scores
    """
    all_scores = defaultdict(list)
    for i, (X, y) in enumerate(tqdm(loader, desc="[----] ")):
        y = y['label']

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

    return all_scores

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")     
    parser.add_argument("--restore-from", type=str, default="",
                    help="Model from which to restore from") 
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

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_NAME = get_save_folder()
    n_channels = 3 if SAVE_NAME == "ImageNet" else 1
    
    numpy.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Loads backbone model
    backbone, cfg = get_pretrained_model_v2(
        name=args.backbone, 
        weights=args.backbone_weights,
        as_classifier=False,
        blocks='all',
        path=None,
        mask_ratio=0.0,
        pretrained=True if SAVE_NAME=="ImageNet" else 1,
        in_channels=n_channels
        )
    cfg.freeze_backbone = True
    update_cfg(cfg, args.opts)

    # Loads dataset and dataset-specific configuration
    _, _, testing_dataset = get_dataset(
        name=args.dataset, 
        cfg=cfg,
        n_channels=n_channels
    )

    cfg.batch_size = 32

    # Loads checkpoint
    checkpoint = torch.load(args.restore_from)
    print(checkpoint.keys())
    OUTPUT_FOLDER = os.path.dirname(args.restore_from)

    # Build the UNet model.
    model = get_decoder(backbone, cfg, in_channels=n_channels, out_channels=1)
    ckpt = checkpoint.get("model", None)
    if not ckpt is None:
        print("Restoring model...")
        model.load_state_dict(ckpt)
    else:
        raise ValueError
    model = model.to(DEVICE)

    # Build a PyTorch dataloader.
    test_loader = torch.utils.data.DataLoader(
        testing_dataset,  # Pass the dataset to the dataloader.
        batch_size=cfg.batch_size,  # A large batch size helps with the learning.
        shuffle=True,  # Shuffling is important!
        num_workers=4
    )
    
    # Puts the model in evaluation mode
    model.eval()
    scores = evaluate_segmentation(model, test_loader)

    # savedir = f"./results/{args.backbone}_{SAVE_NAME}/{args.dataset}/{os.path.basename(OUTPUT_FOLDER)}"
    savedir = f"./results/{args.backbone}_{SAVE_NAME}/{args.dataset}"
    os.makedirs(savedir, exist_ok=True)
    for key, values in scores.items():
        print("Results for", key)
        values = numpy.array(values)
        
        fig, ax = pyplot.subplots(figsize=(3, 3))
        for i in range(values.shape[1]):
            data = values[:, i]
            
            # Remove -1 values as they are not valid
            data = data[data != -1]

            # print(testing_dataset.classes[i])
            print( 
                  "avg : {:0.4f}".format(numpy.mean(data, axis=0)), 
                  "std : {:0.4f}".format(numpy.std(data, axis=0)),
                  "med : {:0.4f}".format(numpy.median(data, axis=0)),)

            bplot = ax.boxplot(data, positions=[i], widths=0.8)
            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                pyplot.setp(bplot[element], color='black')

        ax.set(
            xticks = numpy.arange(values.shape[1]), xticklabels =['synaptic-proteins'],
            ylim = (0, 1)
        )
        pyplot.savefig(os.path.join(savedir, f"{key}.pdf"), bbox_inches="tight", transparent=True)