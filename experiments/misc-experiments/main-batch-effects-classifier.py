
import matplotlib.lines
import numpy
import pandas
import anndata
import warnings
import random
import torch
import argparse
import sys
import os
import json
import matplotlib

from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from phate import PHATE
from umap import UMAP
from matplotlib import pyplot
from scipy.spatial.distance import cdist
from skimage.util import montage
from skimage.io import imsave
from torchvision.transforms.functional import gaussian_blur

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')

    import scanpy
    from scib import metrics

from tiffwrapper import make_composite
from batch_effects import IdendityBatchEffect, RotationBatchEffect, FlipBatchEffect, PoissonBatchEffect, GaussianBatchEffect, ContrastBatchEffect, GaussianBlurBatchEffect

from stedfm.DEFAULTS import BASE_PATH
from stedfm.loaders import get_dataset
from stedfm.model_builder import get_pretrained_model_v2, get_classifier_v3
from stedfm.utils import update_cfg, savefig

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--model", type=str, default='mae-lightning-small')
parser.add_argument("--probe", type=str, default='linear-probe.pth')
parser.add_argument("--pretraining", type=str, default="STED")
parser.add_argument("--opts", nargs="+", default=[], 
                    help="Additional configuration options")    
args = parser.parse_args()

# Assert args.opts is a multiple of 2
if len(args.opts) == 1:
    args.opts = args.opts[0].split(" ")
assert len(args.opts) % 2 == 0, "opts must be a multiple of 2"

cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    name="nice-prism",
    colors=["#5F4690","#1D6996","#38A6A5","#0F8554","#73AF48","#EDAD08","#E17C05","#CC503E","#94346E"]
)
matplotlib.colormaps.register(cmap=cmap, force=True)
matplotlib.colormaps.register(cmap=cmap.reversed(), force=True)

def set_seeds():
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)

class PoissonBatchEffect:
    def __init__(self, _lambda):
        self._lambda = _lambda

    @property
    def name(self):
        return f"PoissonBatchEffect({self._lambda})"

    def apply(self, X):
        noise = torch.poisson(torch.ones_like(X) * self._lambda) / 255.0
        return X + noise
    
class GaussianBatchEffect:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    @property
    def name(self):
        return f"GaussianBatchEffect({self.mean}, {self.std})"

    def apply(self, X):
        noise = torch.normal(mean=self.mean, std=self.std, size=X.size())
        if X.size(0) == 1:
            return torch.clamp(X + noise, 0, None)
        return X + noise
    
class ContrastBatchEffect:
    def __init__(self, cutoff, gain=10):
        self.cutoff = cutoff
        self.gain = gain

    @property
    def name(self):
        return f"ContrastBatchEffect({self.cutoff}, {self.gain})"

    def apply(self, X):
        vmin, vmax = X.min(), X.max()
        X = (X - vmin) / (vmax - vmin)
        output = 1 / (1 + torch.exp(self.gain * (self.cutoff - X)))
        return torch.clamp(output * (vmax - vmin) + vmin, vmin, vmax)

class ScaleBatchEffect:
    def __init__(self, scale):
        self.scale = scale

    @property
    def name(self):
        return f"ScaleBatchEffect({self.scale})"

    def apply(self, X):
        return X * self.scale
    
class OffsetBatchEffect:
    def __init__(self, offset):
        self.offset = offset

    @property
    def name(self):
        return f"OffsetBatchEffect({self.offset})"

    def apply(self, X):
        return X + self.offset
    
class GaussianBlurBatchEffect:
    def __init__(self, sigma):
        self.sigma = sigma

    @property
    def name(self):
        return f"GaussianBlurBatchEffect({self.sigma})"

    def apply(self, X):
        return gaussian_blur(X, kernel_size=5, sigma=self.sigma)
    
class IdendityBatchEffect:
    @property
    def name(self):
        return "IdendityBatchEffect()"

    def apply(self, X):
        return X
    
def get_predictions(model, loader, batch_effects, device="cpu"):
    """
    Get the predictions from the model.

    :param model: Model
    :param loader: DataLoader
    :return: Anndata object
    """
    predictions, labels, augmentations = [], [], []
    
    model.eval()
    with torch.no_grad():

        for batch_effect in tqdm(batch_effects, desc="Batch effects"):
            for X, y in tqdm(loader, desc="Loader", leave=False):
                
                # Apply the batch effect
                X = batch_effect.apply(X)

                X = X.to(device)
                preds, _ = model.forward(X)
                predictions.append(preds.cpu().numpy())
                labels.append(y["label"].cpu().numpy())
                augmentations.append([batch_effect.name] * X.size(0))

    predictions = numpy.concatenate(predictions, axis=0)
    labels = numpy.concatenate(labels, axis=0)
    augmentations = numpy.concatenate(augmentations, axis=0)
    return predictions, labels, augmentations

def get_scores(predictions, labels, augmentations, batch_effects=None):
    """
    Get the scores.

    :param predictions: Predictions
    :param labels: Labels
    :param augmentations: Augmentations
    :return: Dictionary
    """
    predictions = predictions.argmax(axis=-1)

    scores = dict()
    scores["confusion-matrix"] = confusion_matrix(predictions, labels).tolist()
    scores["accuracy"] = accuracy_score(predictions, labels)
    scores["augmentations"] = dict()

    if batch_effects is None:
        batch_effects = numpy.unique(augmentations)

    for effect in batch_effects:
        mask = augmentations == effect
        scores["augmentations"][effect] = dict()
        scores["augmentations"][effect]["confusion-matrix"] = confusion_matrix(predictions[mask], labels[mask]).tolist()
        scores["augmentations"][effect]["accuracy"] = accuracy_score(predictions[mask], labels[mask])
    return scores

def show_examples(effect, images):
    """
    Show the images.

    :param images: List of images
    :return: None
    """
    arr_in = []
    for i in range(len(images)):
        vmin = min([image.min() for image in images[i]]).item()
        vmax = max([image.max() for image in images[i]]).item()
        for image in images[i]:
            image = image[0].numpy()
            arr_in.append(numpy.clip((image - vmin) / (vmax - vmin), 0, 1))
    output = montage(arr_in, grid_shape=(len(images), len(images[0])), padding_width=1, fill=min([im.min() for im in arr_in]))
    imsave(
        os.path.join("results", f"{args.dataset}_{effect}_examples.png"), 
        make_composite(output[numpy.newaxis], ["hot"])
    )

def main():

    train_loader, _, _ = get_dataset(
        name=args.dataset, training=True
    )
    num_classes = train_loader.dataset.num_classes
    print("=====================================")
    print(f"Dataset: {args.dataset}")
    print(f"Num. Classes: {num_classes}")
    print(f"Classes: {train_loader.dataset.classes}")
    print("=====================================")

    set_seeds()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_channels = 3 if "imagenet" in args.pretraining.lower() else 1

    model, cfg = get_classifier_v3(
        name=args.model,
        dataset=args.dataset,
        pretraining=args.pretraining,
        num_classes=num_classes,
        in_channels=n_channels,
        blocks="all",
        probe=args.probe
    )
    model = model.to(device)

    # Update configuration
    cfg.args = args
    update_cfg(cfg, args.opts)

    # Load the dataset
    train_loader, valid_loader, test_loader = get_dataset(
        name=args.dataset,
        transform=None,
        path=None,
        n_channels=n_channels,
        batch_size=cfg.batch_size,
        training=True
    )

    # histogram; 
    batch_effects = {
        "geometric" : [
            IdendityBatchEffect(),
            RotationBatchEffect(),
            FlipBatchEffect(),
        ],
        "poisson" : [
            IdendityBatchEffect(),
            PoissonBatchEffect(1.0),
            PoissonBatchEffect(2.0),
            PoissonBatchEffect(3.0),
            PoissonBatchEffect(4.0),
            PoissonBatchEffect(5.0),
        ],
        "gaussian-noise" : [
            IdendityBatchEffect(),
            GaussianBatchEffect(0.0, 0.01),
            GaussianBatchEffect(0.0, 0.025),
            GaussianBatchEffect(0.0, 0.05),
        ],
        # "contrast" : [
        #     IdendityBatchEffect(),
        #     ContrastBatchEffect(0.30, 10.0),
        #     ContrastBatchEffect(0.40, 10.0),
        #     ContrastBatchEffect(0.50, 10.0),
        # ],
        "gaussian-blur" : [
            IdendityBatchEffect(),
            GaussianBlurBatchEffect(0.1),
            GaussianBlurBatchEffect(0.5),
            GaussianBlurBatchEffect(1.0),
            GaussianBlurBatchEffect(2.0),
        ],
        "mixed" : [
            IdendityBatchEffect(),
            RotationBatchEffect(),
            PoissonBatchEffect(1.0),
            GaussianBatchEffect(0.0, 0.01),
            GaussianBlurBatchEffect(1.0),
        ]
    }

    # Save the results
    os.makedirs("results", exist_ok=True)
    scores = dict()
    for name, effects in batch_effects.items():

        # images = []
        # rng = numpy.random.default_rng(seed=args.seed)
        # choices = rng.choice(len(train_loader.dataset), 5)
        # for idx in choices:
        #     imgs = []
        #     for effect in effects:
        #         imgs.append(effect.apply(train_loader.dataset[idx][0]))
        #     images.append(imgs)
        # show_examples(name, images)
        
        predictions, labels, augmentations = get_predictions(model, test_loader, batch_effects=effects, device=device)
        scores[name] = get_scores(predictions, labels, augmentations, batch_effects=[effect.name for effect in effects])

    os.makedirs(os.path.join("results", "batch-effects-classifier"), exist_ok=True)
    with open(os.path.join("results", "batch-effects-classifier", f"{args.dataset}_{args.model}_{args.pretraining}.json"), "w") as f:
        json.dump(scores, f, indent=4)

if __name__ == "__main__":
    main()
