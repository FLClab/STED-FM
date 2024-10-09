
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
from lightly.data import LightlyDataset
from lightly.models.modules import heads
from tqdm import tqdm
from collections import defaultdict
from collections.abc import Mapping
from multiprocessing import Manager
from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchsummary import summary
from matplotlib import pyplot
from sklearn.decomposition import PCA
from tqdm.auto import trange, tqdm

from pysted import base, utils
from pysted.data import DatamapLoader
from nanopyx.core.analysis.frc import FIRECalculator

import sys 
sys.path.insert(0, "..")

from model_builder import get_base_model, get_pretrained_model_v2
from utils import update_cfg, save_cfg

# Change these values at your own risk
DEFAULTS = {
    "excitation" : base.GaussianBeam(
        635e-9),
    "sted" : base.DonutBeam(
        775e-9, zero_residual=0.01, rate=40e6, tau=400e-12, anti_stoke=False),
    "detector" : base.Detector(
        noise=True, det_delay=750e-12, det_width=8e-9, background=1/50e-6),
    "objective" : base.Objective(
        transmission = {488: 0.84, 535: 0.85, 550: 0.86, 585: 0.85, 590: 0.85, 575: 0.85, 590: 0.85, 635: 0.84, 690: 0.82, 750: 0.77, 775: 0.75}),
    "fluo" : base.Fluorescence(**{
        "lambda_": 6.9e-7,
        "qy": 0.65,
        "sigma_abs": {
            635: 2.14e-20,
            775: 3.5e-25
        },
        "sigma_ste": {
            775: 3.0e-22
        },
        "tau": 3.5e-9,
        "tau_vib": 1e-12,
        "tau_tri": 0.0000012,
        "k0": 0,
        "k1": 1.3e-15,
        "b": 1.6,
        "triplet_dynamics_frac": 0
    })
}

action_spaces = {
    "p_sted" : {"low" : 0., "high" : 350e-3},
    "p_ex" : {"low" : 0., "high" : 10e-6},
    "pdt" : {"low" : 1.0e-6, "high" : 150.0e-6},
}

# Example values of parameters used when doing a STED acquisition
sted_params = {
    "pdt": action_spaces["pdt"]["low"] * 2,
    "p_ex": action_spaces["p_ex"]["high"] * 0.6,
    "p_sted": action_spaces["p_sted"]["high"] * 0.6
}

# Example values of parameters used when doing a Confocal acquisition. Confocals always have p_sted = 0
conf_params = {
    "pdt": action_spaces["pdt"]["low"],
    "p_ex": action_spaces["p_ex"]["high"] * 0.6,
    "p_sted": 0.0
}

pixelsize = 20e-9
microscope = base.Microscope(**DEFAULTS, load_cache=True)
i_ex, i_sted, _ = microscope.cache(pixelsize, save_cache=True)

def acquire(molecules, p_ex=0.3, p_sted=0.0, pdt=0.25, show=True):
    """
    Creates a datamap and acquires an image

    :param p_ex: A `float` of the excitation
    :param p_sted: A `float` of the depletion
    :param pdt: A `float` of the dwelltime
    :param show: A `bool` whether to show the acquired images.
    """
    # acquisitions = defaultdict(list)
    # molecules = create_datamap()

    # Create datamap
    datamap = base.Datamap(molecules, pixelsize)
    datamap.set_roi(i_ex, "max") # This sets the entire datamap as the field of view

    # Defines the imaging parameters
    sted_params = {
        "pdt": (action_spaces["pdt"]["high"] - action_spaces["pdt"]["low"]) * pdt + action_spaces["pdt"]["low"],
        "p_ex": action_spaces["p_ex"]["high"] * p_ex,
        "p_sted": action_spaces["p_sted"]["high"] * p_sted
    }
    conf_params = {
        "pdt": 10e-6,
        "p_ex": 5e-6,
        "p_sted": 0.0
    }

    conf1, _, _ = microscope.get_signal_and_bleach(datamap, datamap.pixelsize, **conf_params,
                                                                bleach=False, update=True, seed=42)
    # Two images are required to use the FRC
    sted_image1, _, _ = microscope.get_signal_and_bleach(datamap, datamap.pixelsize, **sted_params,
                                                                bleach=False, update=True, seed=42)
    sted_image, _, _ = microscope.get_signal_and_bleach(datamap, datamap.pixelsize, **sted_params,
                                                                bleach=False, update=True, seed=43)

    conf2, _, _ = microscope.get_signal_and_bleach(datamap, datamap.pixelsize, **conf_params,
                                                                bleach=False, update=True, seed=42)

    frc_calculator = FIRECalculator(pixel_size=pixelsize * 1e+9, units="nm")
    frc_calculator.calculate_fire_number(sted_image1, sted_image)
    acquisitions["resolution"].append(frc_calculator.fire_number)

    conf_fg = (conf1 > numpy.quantile(conf1, 0.95)).astype(bool)
    photobleaching = (numpy.mean(conf1[conf_fg]) - numpy.mean(conf2[conf_fg])) / numpy.mean(conf1[conf_fg])
    acquisitions["photobleaching"].append(photobleaching)

    acquisitions["parameters"].append(sted_params)
    acquisitions["images"].append({
        "conf1" : conf1,
        "conf2" : conf2,
        "sted" : sted_image
    })

    # if show:
    #     fig, axes = pyplot.subplots(1, 3, figsize=(10, 3))
    #     axes[0].imshow(conf1, cmap="hot", vmin=0, vmax=numpy.quantile(conf1, 0.999))
    #     axes[1].imshow(sted_image, cmap="hot", vmin=0, vmax=numpy.quantile(sted_image, 0.999))
    #     axes[2].imshow(conf2, cmap="hot", vmin=0, vmax=numpy.quantile(conf1, 0.999))

    #     for ax in axes.ravel():
    #         ax.get_xaxis().set_visible(False)
    #         ax.get_yaxis().set_visible(False)
    #         pyplot.colorbar(ax.get_images()[0], ax=ax)

    #     # Sets the title and annotations
    #     axes[0].set_title("CONF1")
    #     axes[1].set_title("STED")
    #     axes[2].set_title("CONF2")
    #     t = axes[1].annotate(
    #         "R: {:0.2f}\nP: {:0.2f}".format(acquisitions["resolution"][-1], acquisitions["photobleaching"][-1]),
    #         (5, 5), horizontalalignment="left", verticalalignment="top",

    #     )
    #     t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white'))
    #     pyplot.show()
    return acquisitions

def plot_features(features, savename, **kwargs):
    if len(features) <= 1:
        return
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(features)

    fig, ax = pyplot.subplots()
    for i in range(len(features)):
        ax.annotate(str(i), (pca_features[i, 0], pca_features[i, 1]))

    # Handles different markers per embedding
    marker = kwargs.pop("marker", "o")
    if isinstance(marker, str):
        marker = [marker] * len(features)
    marker = numpy.array(marker)

    # Handles different colors per embedding
    color = kwargs.pop("color", "k")
    if isinstance(color, str):
        color = [color] * len(features)
    color = numpy.array(color)

    for unique in numpy.unique(marker):
        mask = marker == unique
        ax.scatter(pca_features[mask, 0], pca_features[mask, 1], marker=unique, color=color[mask], **kwargs)
    fig.savefig(savename)
    pyplot.close()

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
    if args.backbone_weights:
        backbone, cfg = get_pretrained_model_v2(args.backbone, weights=args.backbone_weights)
    else:
        backbone, cfg = get_base_model(args.backbone)
    
    model = ForwardModel(args.backbone, backbone)

    all_embeddings, all_cols, all_markers = [], [], []
    MARKERS = {
        "psd95": "o",
        "factin": "s",
        "tubulin": "^"
    }
    for protein in ["psd95", "factin", "tubulin"]:
        loader = DatamapLoader(protein)
        # print(loader[0].shape)
        # out = backbone.forward(loader[0])
        # print(out.shape)

        acquisitions = defaultdict(list)

        cmap = pyplot.get_cmap("tab10")
        embeddings, cols = [], []
        for n in trange(len(loader)):
            for p_sted in tqdm(numpy.linspace(0, 1, 10), desc="STED power", leave=False):
                acquire(loader[n] * 5, p_sted=p_sted)

                sted_image = acquisitions["images"][-1]["sted"]
                img = torch.tensor(sted_image / sted_image.max(), dtype=torch.float32)
                if "imagenet" in args.backbone_weights.lower():
                    img = numpy.tile(img[numpy.newaxis], (3, 1, 1))
                    img = numpy.moveaxis(img, 0, -1)
                    img = transforms.ToTensor()(img)
                    img = transforms.Normalize(mean=[0.0695771782959453, 0.0695771782959453, 0.0695771782959453], std=[0.12546228631005282, 0.12546228631005282, 0.12546228631005282])(img)
                    img = torch.unsqueeze(img, 0)
                else:
                    img = torch.unsqueeze(img, 0).unsqueeze(0)
                
                out = model.forward(img)
                embeddings.append(out.cpu().data.numpy().ravel())
                all_embeddings.append(embeddings[-1])

                cols.append(cmap(n))
                all_cols.append(cols[-1])
                all_markers.append(MARKERS[protein])
                plot_features(embeddings, f"sted-features-{protein}-{args.backbone_weights}.png", color=cols)

                plot_features(all_embeddings, f"sted-features-all-{args.backbone_weights}.png", color=all_cols, marker=all_markers)

