
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
from sklearn.model_selection import GridSearchCV

import sys 
sys.path.insert(0, "..")

from loaders import get_dataset
from model_builder import get_base_model, get_pretrained_model_v2
from utils import update_cfg, save_cfg

class ForwardModel(torch.nn.Module):
    def __init__(
        self,
        name:str,
        backbone: torch.nn.Module,
        global_pool: str = "avg",
    ) -> None:
        super().__init__()
        self.name = name
        self.backbone = backbone
        self.global_pool = global_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if "mae" in self.name.lower():
            features = self.backbone.forward_encoder(x)
            if self.global_pool == "token":
                features = features[:, 0, :] # class token 
            elif self.global_pool == "avg":
                features = torch.mean(features[:, 1:, :], dim=1) # Average patch tokens
            else:
                raise NotImplementedError(f"Invalid `{self.global_pool}` pooling function.")
        else:
            features = self.backbone.forward(x)
        return features

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")     
    parser.add_argument("--backbone", type=str, default="resnet18",
                        help="Backbone model to load")
    parser.add_argument("--backbone-weights", type=str, default=None,
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
    if args.backbone_weights in (None, "null", "None", "none"):
        args.backbone_weights = None

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    numpy.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Loads backbone model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.backbone_weights:
        backbone, cfg = get_pretrained_model_v2(args.backbone, weights=args.backbone_weights)
    else:
        backbone, cfg = get_base_model(args.backbone)
    model = ForwardModel(args.backbone, backbone)
    model = model.to(DEVICE)

    train_loader, valid_loader, test_loader = get_dataset(
        "optim", training=True, n_channels=cfg.in_channels, 
        batch_size=32, seed=args.seed, transforms=None,
        min_quality_score=0.,
    )

    X_train, y_train = [], []
    for batch in train_loader:
        images, labels = batch
        # print(labels["score"])

        images = images.to(DEVICE)
        features = model(images)
        features = features.detach().cpu().numpy()
        X_train.extend(features)
        y_train.extend(labels["score"].numpy())

    X_train = numpy.array(X_train)
    noise_level = numpy.mean(numpy.std(X_train, axis=0)) * 0.1
    y_train = numpy.array(y_train)

    print(X_train.shape, y_train.shape)
    print("Training GP...")

    # parameters = {
    #     "kernel": [kernels.WhiteKernel(noise_level=noise_level, noise_level_bounds=(0.001 * noise_level, noise_level * 10)) + kernels.RBF(length_scale=l) for l in [0.1, 1.0, 5.0, 10.0]],
    # }
    # clf = GridSearchCV(
    #     GaussianProcessRegressor(random_state=args.seed), 
    #     parameters, verbose=10, cv=3, 
    #     scoring="r2", refit=True)
    # clf.fit(X_train, y_train)

    # gp = clf.best_estimator_

    gp = GaussianProcessRegressor(
        kernel=kernels.WhiteKernel(noise_level=noise_level, noise_level_bounds=(0.001 * noise_level, noise_level * 10)) + kernels.RBF(length_scale=1.0), 
        random_state=args.seed)
    gp.fit(X_train, y_train)

    print("Testing GP...")

    X_test, y_test, all_images = [], [], []
    for batch in test_loader:
        images, labels = batch
        all_images.extend(images.numpy())
        images = images.to(DEVICE)
        features = model(images)
        features = features.detach().cpu().numpy()
        X_test.extend(features)
        y_test.extend(labels["score"].numpy())

    X_test = numpy.array(X_test)
    y_test = numpy.array(y_test)

    all_images = numpy.array(all_images)
    m, M = all_images.min(), all_images.max()
    all_images = (all_images - m) / (M - m)

    y_pred, y_pred_std = gp.predict(X_test, return_std=True)
    print(y_pred_std.min(), y_pred_std.max())

    fig, ax = pyplot.subplots()
    ax.errorbar(y_test, y_pred, yerr=y_pred_std, fmt='o', alpha=0.5)
    ax.set(
        xlabel="True quality score",
        ylabel="Predicted quality score",
        ylim=(0, 1),
        xlim=(0, 1),
    )
    fig.savefig(f"gp-{args.backbone_weights}.png")

    argsort = numpy.argsort(y_pred_std)
    fig, axes = pyplot.subplots(2, 7, figsize=(10, 5), tight_layout=True)
    for i, ax in enumerate(axes[0].flatten()):
        idx = argsort[i]
        ax.imshow(all_images[idx].transpose(1, 2, 0), cmap="gray", vmin=0, vmax=0.7 * all_images[idx].max())
        ax.set_title(f"True: {y_test[idx]:.2f}, Pred: {y_pred[idx]:.2f}", fontsize=8)
        ax.axis("off")
    for i, ax in enumerate(axes[1].flatten()):
        idx = argsort[-i-1]
        ax.imshow(all_images[idx].transpose(1, 2, 0), cmap="gray", vmin=0, vmax=0.7 * all_images[idx].max())
        ax.set_title(f"True: {y_test[idx]:.2f}, Pred: {y_pred[idx]:.2f}", fontsize=8)
        ax.axis("off")
    fig.savefig(f"gp-samples-{args.backbone_weights}.png", dpi=300)

    pred_difference = numpy.abs(y_pred - y_test)
    argsort = numpy.argsort(pred_difference)
    fig, axes = pyplot.subplots(2, 7, figsize=(10, 5), tight_layout=True)
    for i, ax in enumerate(axes[0].flatten()):
        idx = argsort[i]
        ax.imshow(all_images[idx].transpose(1, 2, 0), cmap="gray", vmin=0, vmax=0.7 * all_images[idx].max())
        ax.set_title(f"True: {y_test[idx]:.2f}, Pred: {y_pred[idx]:.2f}", fontsize=8)
        ax.axis("off")
    for i, ax in enumerate(axes[1].flatten()):
        idx = argsort[-i - 1]
        ax.imshow(all_images[idx].transpose(1, 2, 0), cmap="gray", vmin=0, vmax=0.7 * all_images[idx].max())
        ax.set_title(f"True: {y_test[idx]:.2f}, Pred: {y_pred[idx]:.2f}", fontsize=8)
        ax.axis("off")
    fig.savefig(f"gp-samples-difference-{args.backbone_weights}.png", dpi=300)