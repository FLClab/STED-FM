
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
from matplotlib import pyplot
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr

import sys 
sys.path.insert(0, "..")

from DEFAULTS import COLORS
from loaders import get_dataset
from model_builder import get_base_model, get_pretrained_model_v2
from utils import update_cfg, save_cfg, savefig

def set_seeds():
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)

def get_features(model, loader):
    """
    Runs the model on the loader and returns the features and labels

    :param model: the model to run
    :param loader: the loader to run the model on

    :returns: a tuple with the features, labels, and processed images
    """
    model.eval()
    X, y, images = [], [], []
    for batch in tqdm(loader, desc="Extracting features"):
        x, labels = batch
        images.extend(x.numpy())
        x = x.to(device)
        features = model.forward_features(x)
        features = features.detach().cpu().numpy()
        X.extend(features)
        y.extend(labels["score"].numpy())
    return numpy.array(X), numpy.array(y), numpy.array(images)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")     
    parser.add_argument("--model", type=str, default="resnet18",
                        help="model model to load")
    parser.add_argument("--weights", type=str, default=None,
                        help="Backbone model to load")
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
    if args.weights in (None, "null", "None", "none"):
        args.weights = None

    set_seeds()

    # Loads backbone model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_channels = 1
    if args.weights is not None:
        n_channels = 3 if "imagenet" in args.weights.lower() else n_channels

    model, cfg = get_pretrained_model_v2(
        name=args.model,
        weights=args.weights,
        path=None,
        mask_ratio=0.0, 
        pretrained=True if n_channels==3 else False,
        in_channels=n_channels,
        as_classifier=True,
        blocks="all",
        num_classes=1
    )
    model = model.to(device)

    # Update configuration
    cfg.args = args
    update_cfg(cfg, args.opts)

    train_loader, valid_loader, test_loader = get_dataset(
        "optim", training=True, n_channels=cfg.in_channels, 
        batch_size=cfg.batch_size, seed=args.seed, transforms=None,
        min_quality_score=0.,
    )

    random.seed(42)
    choices = random.sample(range(len(test_loader.dataset)), 50)
    cmap = pyplot.get_cmap("gray")
    for choice in choices:
        img, metadata = test_loader.dataset[choice]
        print(metadata)
        img = img.numpy()[0]

        img = cmap(img)[:, :, :-3] * 255.
        import tifffile 
        os.makedirs("tmp/optim", exist_ok=True)
        tifffile.imwrite(f"tmp/optim/{metadata['label']}-{metadata['dataset-idx']}.tif", img.astype(numpy.uint8))

    exit()

    X_train, y_train, _ = get_features(model, train_loader)
    X_valid, y_valid, _ = get_features(model, valid_loader)
    X_test, y_test, _ = get_features(model, test_loader)

    savedir = os.path.join(".", "features", "resolution")
    os.makedirs(savedir, exist_ok=True)
    numpy.savez(os.path.join(savedir, f"features-{args.weights}.npz"), X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, X_test=X_test, y_test=y_test)

    noise_level = numpy.mean(numpy.std(X_train, axis=0)) * 0.1

    print(X_train.shape, y_train.shape)

    print("Training regression model...")

    clf = Ridge(random_state=args.seed)
    clf.fit(X_train, y_train)

    print("Testing regression model...")

    X_test, y_test, all_images = get_features(model, test_loader)

    # Normalize images
    m, M = all_images.min(), all_images.max()
    all_images = (all_images - m) / (M - m)

    y_pred = clf.predict(X_test)

    os.makedirs(os.path.join("figures", "quality"), exist_ok=True)

    fig, ax = pyplot.subplots(figsize=(3, 3))
    ax.scatter(y_test, y_pred, alpha=0.5, color=COLORS[args.weights])
    ax.plot([0, 1], [0, 1], color="black", linestyle="--")
    pearson, stats = pearsonr(y_test, y_pred)
    ax.annotate(f"Pearson: {pearson:.2f}", (0.05, 0.95), xycoords="axes fraction", ha="left", va="top")
    ax.set(
        xlabel="True quality score",
        ylabel="Predicted quality score",
        ylim=(0, 1),
        xlim=(0, 1),
    )
    savefig(fig, os.path.join("figures", "quality", f"{args.model}_{args.weights}_scatter"), save_white=True)

    # argsort = numpy.argsort(y_pred_std)
    # fig, axes = pyplot.subplots(2, 7, figsize=(10, 5), tight_layout=True)
    # for i, ax in enumerate(axes[0].flatten()):
    #     idx = argsort[i]
    #     ax.imshow(all_images[idx].transpose(1, 2, 0), cmap="gray", vmin=0, vmax=0.7 * all_images[idx].max())
    #     ax.set_title(f"True: {y_test[idx]:.2f}, Pred: {y_pred[idx]:.2f}", fontsize=8)
    #     ax.axis("off")
    # for i, ax in enumerate(axes[1].flatten()):
    #     idx = argsort[-i-1]
    #     ax.imshow(all_images[idx].transpose(1, 2, 0), cmap="gray", vmin=0, vmax=0.7 * all_images[idx].max())
    #     ax.set_title(f"True: {y_test[idx]:.2f}, Pred: {y_pred[idx]:.2f}", fontsize=8)
    #     ax.axis("off")
    # fig.savefig(f"gp-samples-{args.weights}.png", dpi=300)

    # pred_difference = numpy.abs(y_pred - y_test)
    # argsort = numpy.argsort(pred_difference)
    # fig, axes = pyplot.subplots(2, 7, figsize=(10, 5), tight_layout=True)
    # for i, ax in enumerate(axes[0].flatten()):
    #     idx = argsort[i]
    #     ax.imshow(all_images[idx].transpose(1, 2, 0), cmap="gray", vmin=0, vmax=0.7 * all_images[idx].max())
    #     ax.set_title(f"True: {y_test[idx]:.2f}, Pred: {y_pred[idx]:.2f}", fontsize=8)
    #     ax.axis("off")
    # for i, ax in enumerate(axes[1].flatten()):
    #     idx = argsort[-i - 1]
    #     ax.imshow(all_images[idx].transpose(1, 2, 0), cmap="gray", vmin=0, vmax=0.7 * all_images[idx].max())
    #     ax.set_title(f"True: {y_test[idx]:.2f}, Pred: {y_pred[idx]:.2f}", fontsize=8)
    #     ax.axis("off")
    # fig.savefig(f"gp-samples-difference-{args.weights}.png", dpi=300)