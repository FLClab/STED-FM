import torch 
import os
import argparse 
import matplotlib.pyplot as plt
import sys 
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm 
from torchvision import transforms
import numpy as np
import seaborn 
from scipy.spatial.distance import pdist, cdist
import pandas 
import random

from tiffwrapper import make_composite

sys.path.insert(0, "../")
from loaders import get_dataset 
from model_builder import get_pretrained_model_v2 
from utils import SaveBestModel, AverageMeter, compute_Nary_accuracy, track_loss, update_cfg, get_number_of_classes
from modules.relax import RELAX

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--dataset", type=str, default="synaptic-proteins")
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--global-pool", type=str, default="avg")
parser.add_argument("--opts", nargs="+", default=[], 
                    help="Additional configuration options")
args = parser.parse_args()

# Assert args.opts is a multiple of 2
if len(args.opts) == 1:
    args.opts = args.opts[0].split(" ")
assert len(args.opts) % 2 == 0, "opts must be a multiple of 2"

def set_seeds():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def get_save_folder() -> str: 
    if "imagenet" in args.weights.lower():
        return "ImageNet"
    elif "sted" in args.weights.lower():
        return "STED"
    elif "jump" in args.weights.lower():
        return "JUMP"
    elif "ctc" in args.weights.lower():
        return "CTC"
    elif "hpa" in args.weights.lower():
        return "HPA"
    else:
        raise NotImplementedError("The requested weights do not exist.")
    
def show_image(image, relax):

    image = image.squeeze().cpu().data.numpy()
    if image.ndim == 3:
        image = image[0]
    importance = relax.importance().squeeze().cpu().data.numpy()
    uncertainty = relax.uncertainty().squeeze().cpu().data.numpy()

    # importance = np.clip(importance, np.quantile(importance, 0.5), np.quantile(importance, 1.0))
    # uncertainty = np.clip(uncertainty, np.quantile(uncertainty, 0.5), np.quantile(uncertainty, 1.0))

    m, M = np.min(image), np.max(image)
    image_rgb = make_composite(np.array([image]), luts=["grey"], ranges=[(m, M)])
    print(importance.min(), importance.max())
    image_importance_rgb = make_composite(np.stack([image, importance]), luts=["grey", "Orange Hot"], ranges=[(m, M), (importance.min() + 0.5 *(importance.max() - importance.min()), importance.max())])
    image_uncertainty_rgb = make_composite(np.stack([image, uncertainty]), luts=["grey", "Orange Hot"], ranges=[(m, M), (uncertainty.min() + 0.5 *(uncertainty.max() - uncertainty.min()), uncertainty.max())])

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Image")
    axes[1].imshow(image_importance_rgb)
    axes[1].set_title("Importance")
    axes[2].imshow(image_uncertainty_rgb)
    axes[2].set_title("Uncertainty")
    for ax in axes:
        ax.axis("off")
    fig.savefig(f"relax-{args.model}-{args.weights}.png", bbox_inches="tight", dpi=300, facecolor=None)
    plt.close()
    
def run(model, cfg, loader, device, savename, num_masks=3000):
    
    for x, data_dict in loader:

        x = x.to(device)
        with torch.no_grad():
            relax = RELAX(x, model.forward_features, num_batches=round(num_masks / cfg.batch_size), batch_size=cfg.batch_size, verbose=True)
            relax.forward()

        show_image(x, relax)

        # break

def main():
    set_seeds()
    SAVE_NAME = get_save_folder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {device} ---")
    n_channels = 3 if SAVE_NAME == "ImageNet" else 1
    model, cfg = get_pretrained_model_v2(
        name=args.model,
        weights=args.weights,
        as_classifier=True,
        num_classes=1,
        blocks='all',
        global_pool="avg",
        mask_ratio=0,
    ) 

    # Update configuration
    cfg.args = args
    update_cfg(cfg, args.opts)

    # Dataset is iterated one sample at a time
    _, _, test_loader = get_dataset(
        name=args.dataset,
        transform=None,
        training=True,
        path=None,
        n_channels=n_channels,
        batch_size=1,
        num_samples=None,
    )

    model = model.to(device)
    model.eval()

    run(model=model, cfg=cfg, loader=test_loader, device=device, savename=SAVE_NAME)

if __name__=="__main__":
    main()
