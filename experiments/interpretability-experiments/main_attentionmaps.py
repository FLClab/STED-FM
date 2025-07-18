"""
Clarifications:
    Our ViT-small has a depth of 12 (12 blocks), and 6 attention heads per block.
    The code below will show the average attention of the 12 blocks for a given image and a given model, where the average is over the 6 heads for that block.
"""

import timm 
from timm.models.layers import PatchEmbed 
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import argparse
from pprint import pprint
import matplotlib.pyplot as plt
import sys 
import numpy as np
from tiffwrapper import make_composite
import torch.nn.functional as F
import random
import torch
import os
from tqdm import tqdm
import tarfile
from stedfm.DEFAULTS import BASE_PATH
from stedfm.loaders import get_dataset 
from stedfm.model_builder import get_pretrained_model_v2
from stedfm.utils import SaveBestModel, AverageMeter, update_cfg, get_number_of_classes 

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=4242)
parser.add_argument("--dataset", type=str, default="optim")
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--ckpt-path", type=str, default=None)
parser.add_argument("--save-folder", type=str, default="candidates")

parser.add_argument("--opts", nargs="+", default=[])
args = parser.parse_args()

def get_save_folder() -> str: 
    if args.weights is None:
        return "from-scratch"
    elif "imagenet" in args.weights.lower():
        return "ImageNet"
    elif "sted" in args.weights.lower():
        return "STED"
    elif "jump" in args.weights.lower():
        return "JUMP"
    elif "sim" in args.weights.lower():
        return "SIM"
    elif "hpa" in args.weights.lower():
        return "HPA"
    elif "sim" in args.weights.lower():
        return "SIM"
    else:
        raise NotImplementedError("The requested weights do not exist.")


def get_candidates_folder() -> str:
    if args.ckpt_path is None:
        return "candidates"
    elif "None" in args.ckpt_path:
        seed = args.ckpt_path.split("/")[-1].split(".")[0].split("_")[-1]
        return f"candidates_finetuned_seed{seed}"
    else:
        samples = args.ckpt_path.split("/")[-1].split(".")[0].split("_")[1]
        return f"candidates_{samples}samples"

def get_scale_factor() -> float:
    pass 

def save_image(image, a_map, i, folder):
    image = image.squeeze().cpu().data.numpy()
    a_map = a_map.squeeze().cpu().data.numpy()
    if image.ndim == 3:
        image = image[0]

    m, M = np.min(image), np.max(image)
    image_rgb = make_composite(np.array([image]), luts=["gray"], ranges=[(m, M)])
    image_amap_rgb = make_composite(np.stack([image, a_map]), luts=["gray", "Orange Hot"], ranges=[(m, M), (a_map.min() + 0.25 *(a_map.max() - a_map.min()), a_map.max())])

    fig = plt.figure()
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.savefig(f"./attention-map-examples/templates/template{i}.png", dpi=1200, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(image_amap_rgb)
    plt.axis("off")
    plt.savefig(f"./attention-map-examples/{folder}/test_{args.weights}_template{i}.png", dpi=1200, bbox_inches="tight")

    plt.close(fig)

def show_image(image, a_map, i):
    savefolder = "results" if args.ckpt_path is None else "results-finetuned"
    image = image.squeeze().cpu().data.numpy()
    a_map = a_map.squeeze().cpu().data.numpy()
    print(image.shape, a_map.shape)
    if image.ndim == 3:
        image = image[0]

    # importance = np.clip(importance, np.quantile(importance, 0.5), np.quantile(importance, 1.0))
    # uncertainty = np.clip(uncertainty, np.quantile(uncertainty, 0.5), np.quantile(uncertainty, 1.0))

    m, M = np.min(image), np.max(image)
    image_rgb = make_composite(np.array([image]), luts=["gray"], ranges=[(m, M)])
    image_amap_rgb = make_composite(np.stack([image, a_map]), luts=["gray", "Orange Hot"], ranges=[(m, M), (a_map.min() + 0.25 *(a_map.max() - a_map.min()), a_map.max())])

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Image")
    axes[1].imshow(a_map, cmap='viridis')
    axes[1].set_title("Attention\nmap")
    axes[2].imshow(image_amap_rgb)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.savefig(f"./attention-map-examples/{args.weights}_amap_{i}.pdf", bbox_inches="tight", dpi=1200, facecolor=None)
    plt.close()

def set_seeds():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def main():
    np.random.seed(args.seed)
    SAVENAME = get_save_folder()
    CANDIDATES_FOLDER = get_candidates_folder()
    os.makedirs(f"./attention-map-examples/{CANDIDATES_FOLDER}", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {device} ---")
    n_channels =  3 if SAVENAME == "ImageNet" else 1
    _, _, test_loader = get_dataset(name=args.dataset, training=True, n_channels=n_channels)
    test_dataset = test_loader.dataset
    labels = test_dataset.labels 
    uniques = np.unique(labels)
    indices = []
    for u in uniques:
        cls_indices = np.where(labels == u)[0]
        cls_indices = np.random.choice(cls_indices, size=10, replace=False)
        indices.extend(cls_indices)
    

    model, cfg = get_pretrained_model_v2(
        name=args.model,
        weights=args.weights,
        path=None,
        mask_ratio=0.0, 
        pretrained=True if n_channels==3 else False,
        in_channels=n_channels,
        as_classifier=True,
        blocks="all",
        num_classes=4
    )

    if args.ckpt_path is not None:
        print(f"=== Loading checkpoint from {args.ckpt_path} ===")

        checkpoint = torch.load(args.ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)

    print(model)
    model.eval()
    nodes, _ = get_graph_node_names(model.backbone, tracer_kwargs={'leaf_modules': [PatchEmbed]})
    # pprint(nodes)   
    num_blocks = 12
    for i in tqdm(indices, total=len(indices), desc="Processing images"):
        img, _ = test_dataset[i]
        attention_maps = []
    
        img = img.clone().detach().unsqueeze(0).to(device) # B = 1
        with torch.no_grad():
            for n in range(num_blocks):
                feature_extractor = create_feature_extractor(
                    model, return_nodes=[f'backbone.blocks.{n}.attn.q_norm', f'backbone.blocks.{n}.attn.k_norm'],
                    tracer_kwargs={'leaf_modules': [PatchEmbed]})

                out = feature_extractor(img)
                q, k = out[f'backbone.blocks.{n}.attn.q_norm'], out[f'backbone.blocks.{n}.attn.k_norm']
                factor = (384 / 6) ** -0.5 # (head_dim / num_heads ) ** -0.5
                q = q * factor 
                attn = q @ k.transpose(-2, -1) # (1, 6, 197, 197)
                attn = attn.softmax(dim=-1) # (1, 6, 197, 197)
                attn_map = attn.mean(dim=1).squeeze(0)  # (197, 197)
                cls_attn_map = attn[:, :, 0, 1:]  # (1, 6, 196)
                cls_attn_map = cls_attn_map.mean(dim=1).view(14, 14).detach() # (14, 14)
                cls_resized = F.interpolate(cls_attn_map.view(1, 1, 14, 14), (224, 224), mode='bilinear').view(224, 224, 1) # (224, 224, C)
                m, M = cls_resized.min(), cls_resized.max()
                cls_resized = (cls_resized - m) / (M - m)
                attention_maps.append(cls_resized)

            attention_maps = torch.stack(attention_maps)
            attention_map = torch.sum(attention_maps, dim=0)
            save_image(img, attention_map, i=i, folder=CANDIDATES_FOLDER)

if __name__=="__main__":
    main()