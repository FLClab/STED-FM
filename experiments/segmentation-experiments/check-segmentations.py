import numpy as np 
import matplotlib.pyplot as plt 
import os
from typing import List, Tuple, Dict, Any, Optional
import random 
import argparse 
from dataclasses import dataclass 
from tqdm import tqdm 
from collections import defaultdict 
from decoders import get_decoder 
from datasets import get_dataset 
import torch
import sys 
sys.path.insert(0, "../")
from DEFAULTS import BASE_PATH, COLORS 
from configuration import Configuration
from utils import update_cfg
from model_builder import get_base_model, get_pretrained_model_v2 

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ckpt-path", type=str, default=None)
parser.add_argument("--dataset", type=str, default="synaptic-semantic-segmentation")
parser.add_argument("--backbone", type=str, default="mae-lightning-small")
parser.add_argument("--backbone-weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--n-examples", type=int, default=10)
parser.add_argument("--opts", nargs="+", default=[])
args = parser.parse_args()

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_example(img: np.ndarray, pred: np.ndarray, mask: np.ndarray, img_index: int) -> None:
    os.makedirs(f"./results/{args.backbone}/{args.dataset}", exist_ok=True)

    img = img[0]
    n_classes = pred.shape[0]
    vmax = 0
    for i in range(n_classes):
        class_max = pred[i].max()
        if class_max > vmax:
            vmax = class_max


    # Create a figure with subplots
    fig, axs = plt.subplots(2, n_classes+1, figsize=(20, 5))
    for i in range(axs.shape[0]):
        for cls_idx in range(n_classes + 1):
            ax = axs[i, cls_idx]
            if cls_idx == 0:
                ax.imshow(img, cmap='hot', vmin=0, vmax=1)
            else:
                if i == 0:
                    temp = mask[cls_idx-1]
                    ax.imshow(temp, cmap='gray')
                else:
                    temp = pred[cls_idx-1]
                    ax.imshow(temp, cmap='gray', vmin=0, vmax=vmax)
            ax.axis('off') 
    for ax, title in zip(axs[0, :], ["Original", "Round", "Elongated", "Perforated", "Multidomain"]):
        ax.set_title(title)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(f"./results/{args.backbone}/{args.dataset}/{args.backbone_weights}-example_{img_index}.png", dpi=1200)
    plt.close()

class SegmentationConfiguration(Configuration):
    freeze_backbone: bool = True
    num_epochs: int = 300
    learning_rate: float = 1e-4

def main():
    set_seeds(seed=args.seed)
    if len(args.opts) == 1:
        args.opts = args.opts[0].split(" ")
    assert len(args.opts) % 2 == 0, "opts must be a multiple of 2"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert args.backbone_weights is not None, "Backbone weights must be provided"

    backbone, cfg = get_pretrained_model_v2(args.backbone, weights=args.backbone_weights)
    backbone.to(DEVICE)

    training_dataset, validation_dataset, testing_dataset = get_dataset(name=args.dataset, cfg=cfg)
    cfg.args = args 
    segmentation_cfg = SegmentationConfiguration()
    for key, value in segmentation_cfg.__dict__.items():
        setattr(cfg, key, value)

    cfg.backbone_weights = args.backbone_weights 
    print(f"Config: {cfg.__dict__}")
    update_cfg(cfg, args.opts)

    assert args.ckpt_path is not None, "Checkpoint path must be provided"

    model = get_decoder(backbone, cfg)
    checkpoint = torch.load(f"{args.ckpt_path}/result.pt")
    model.load_state_dict(checkpoint["model"])
    model = model.to(DEVICE)
    model.eval()


    N = len(testing_dataset)
    indices = np.random.choice(N, size=args.n_examples, replace=False)

    for i, idx in enumerate(indices):
        img, mask = testing_dataset[idx] 
        img = img.unsqueeze(0).to(DEVICE) 
        pred = model(img)
        img = img.squeeze(0).detach().cpu().numpy()
        pred = pred.squeeze(0).detach().cpu().numpy()
        mask = mask.squeeze(0).detach().cpu().numpy()
        save_example(img=img, pred=pred, mask=mask, img_index=idx)    
        print(f"--- Saved image {i+1} of {args.n_examples} ---")
        

if __name__ == "__main__":
    main()


