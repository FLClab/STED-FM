import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import tarfile
from torch.utils.data import Dataset, DataLoader
import argparse 
from typing import Optional, Callable, Tuple
import io
from skimage.filters import threshold_otsu
from tqdm import tqdm, trange
from tiffwrapper import make_composite
from timm.models.layers import PatchEmbed 
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import sys 
import torch.nn.functional as F
import os
from wavelet import detect_spots

from stedfm.DEFAULTS import BASE_PATH
from stedfm.model_builder import get_pretrained_model_v2
from stedfm.loaders import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="neural-activity-states")
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
args = parser.parse_args()

def save_image(image, a_map, i, mask):
    a_map = a_map.squeeze().cpu().detach().numpy()
    if image.ndim == 3:
        image = image[0]

    m, M = np.min(image), np.max(image)
    image_rgb = make_composite(np.array([image]), luts=["gray"], ranges=[(m, M)])
    image_amap_rgb = make_composite(np.stack([image, a_map]), luts=["gray", "Orange Hot"], ranges=[(m, M), (a_map.min() + 0.25 *(a_map.max() - a_map.min()), a_map.max())])

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(image_rgb)
    axs[1].imshow(a_map, cmap='viridis')
    axs[2].imshow(image_amap_rgb)
    for ax in axs:
        ax.axis("off")
    plt.savefig(f"./zeroshot-examples/{args.weights}_img_{i}.png", dpi=1200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    np.random.seed(42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {DEVICE} ---")
    channels = 3 if "imagenet" in args.weights else 1
    _, _, test_loader = get_dataset(
        name=args.dataset,
        training=True,
        n_channels=channels
    )
    dataset = test_loader.dataset
    model, cfg = get_pretrained_model_v2(
        name=args.model,
        weights=args.weights,
        path=None,
        mask_ratio=0.0, 
        pretrained=True if channels == 3 else False,
        in_channels=channels,
        as_classifier=True,
        blocks="all",
        num_classes=4
    )
    model.to(DEVICE)
    model.eval()
    
    nodes, _ = get_graph_node_names(model.backbone, tracer_kwargs={'leaf_modules': [PatchEmbed]})
    num_blocks = 12
    ratios = []
    save_indices = np.random.choice(range(len(dataset)), size=10, replace=False)
    for i in trange(len(dataset)):
        img, _ = dataset[i]
        temp_img = img.squeeze().cpu().detach().numpy()
        if channels == 3:
            temp_img = temp_img[0]

        if args.dataset == "neural-activity-states":
            mask = detect_spots(temp_img)
        else:
            mask = temp_img > threshold_otsu(temp_img)
        
        img = img.to(DEVICE).unsqueeze(0)
        attention_maps = []
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
            amap = torch.sum(attention_maps, dim=0)
            amap = amap.squeeze().cpu().detach().numpy() 
            sorted_attn = np.sort(amap.flatten())[::-1]
            cumsum_attn = np.cumsum(sorted_attn)
            cumsum_attn /= cumsum_attn[-1]
            threshold_idx = np.searchsorted(cumsum_attn, 0.6)
            threshold_value = sorted_attn[threshold_idx]
            amap = amap > threshold_value

            amap = amap.astype(np.uint8)
            amap_mask = (amap > 0).astype(np.uint8)
            map_hit = (amap[amap_mask] * mask[amap_mask]).sum()
            map_total = amap[amap_mask].sum()
            jaccard_index = map_hit / (map_total)

            # intersection = np.sum(amap & mask)
            # union = np.sum(amap) + np.sum(mask) - intersection
            # jaccard_index = intersection / union if union > 0 else 0
            ratios.append(jaccard_index)
            
    print(f"{args.weights}: {np.mean(ratios)}")


    