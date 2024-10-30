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
import torch
sys.path.insert(0, "../")
from DEFAULTS import BASE_PATH
from loaders import get_dataset 
from model_builder import get_pretrained_model_v2
from utils import SaveBestModel, AverageMeter, update_cfg, get_number_of_classes 

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, default="optim")
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
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
    elif "ctc" in args.weights.lower():
        return "CTC"
    elif "hpa" in args.weights.lower():
        return "HPA"
    else:
        raise NotImplementedError("The requested weights do not exist.")


def get_scale_factor() -> float:
    pass 


def show_image(image, a_map, i):
    image = image.squeeze().cpu().data.numpy()
    a_map = a_map.squeeze().cpu().data.numpy()
    print(image.shape, a_map.shape)
    if image.ndim == 3:
        image = image[0]

    # importance = np.clip(importance, np.quantile(importance, 0.5), np.quantile(importance, 1.0))
    # uncertainty = np.clip(uncertainty, np.quantile(uncertainty, 0.5), np.quantile(uncertainty, 1.0))

    m, M = np.min(image), np.max(image)
    image_rgb = make_composite(np.array([image]), luts=["grey"], ranges=[(m, M)])
    image_amap_rgb = make_composite(np.stack([image, a_map]), luts=["grey", "Orange Hot"], ranges=[(m, M), (a_map.min() + 0.5 *(a_map.max() - a_map.min()), a_map.max())])

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Image")
    axes[1].imshow(a_map, cmap='viridis')
    axes[1].set_title("Attention\nmap")
    axes[2].imshow(image_amap_rgb)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.savefig(f"./{args.weights}_amap_{i}.png", bbox_inches="tight", dpi=1200, facecolor=None)
    plt.close()

def main():
    SAVENAME = get_save_folder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {device} ---")
    n_channels =  3 if SAVENAME == "ImageNet" else 1
    _, _, test_loader = get_dataset(name=args.dataset, training=True, n_channels=n_channels)
    test_dataset = test_loader.dataset
    model, cfg = get_pretrained_model_v2(
        name=args.model,
        weights=args.weights,
        path=None,
        mask_ratio=0.0, 
        pretrained=True if n_channels==3 else False,
        in_channels=n_channels,
        as_classifier=True,
        blocks="0",
        num_classes=4
    )

    nodes, _ = get_graph_node_names(model, tracer_kwargs={'leaf_modules': [PatchEmbed]})
    pprint(nodes)   
    num_blocks = 12
    for i in range(20):
        img, _ = test_dataset[i]
       
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) # B = 1
        with torch.no_grad():
            for n in range(num_blocks):
                feature_extractor = create_feature_extractor(
                    model, return_nodes=[f'backbone.blocks.{n}.attn.q_norm', f'backbone.blocks.{n}.attn.k_norm'],
                    tracer_kwargs={'leaf_modules': [PatchEmbed]})

                out = feature_extractor(img)
                q, k = out[f'backbone.blocks.{n}.attn.q_norm'], out[f'backbone.blocks.{n}.attn.k_norm']
                q = q * 0.1767766952966369 # hard-coded vit-small head_dim --> (head_dim / num_heads ) ** -0.5 = (384 / 12) ** -0.5
                attn = q @ k.transpose(-2, -1) # (1, 6, 197, 197)
                attn = attn.softmax(dim=-1) # (1, 6, 197, 197)
                attn_map = attn.mean(dim=1).squeeze(0)  # (197, 197)
                cls_attn_map = attn[:, :, 0, 1:]  # (1, 6, 196)
                cls_attn_map = cls_attn_map.mean(dim=1).view(14, 14).detach() # (14, 14)
                cls_resized = F.interpolate(cls_attn_map.view(1, 1, 14, 14), (224, 224), mode='bilinear').view(224, 224, 1) # (224, 224, C)
                show_image(img, cls_resized, i=n)
                
            exit()

if __name__=="__main__":
    main()