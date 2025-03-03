from re import L
import numpy as np
import matplotlib.pyplot as plt 
import torch 
from torch.utils.data import DataLoader
import io 
import torch.nn.functional as F 
from datasets import get_dataset 
import argparse 
from tqdm import tqdm, trange
import os
import sys
sys.path.insert(0, '../..')
from DEFAULTS import BASE_PATH, COLORS 
from model_builder import get_pretrained_model_v2 

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, default="synaptic-semantic-segmentation")
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--patch-size", type=int, default=16)
parser.add_argument("--num-patches", type=int, default=14)
parser.add_argument("--image-id", type=int, default=367)
parser.add_argument("--patch-id", type=int, default=78)
parser.add_argument("--num-retrievals", type=int, default=10)
parser.add_argument("--class-id", type=int, default=2)
args = parser.parse_args()

def get_valid_patches():
    valid_patches = np.arange(196)
    remove_1 = np.arange(14)
    remove_2 = np.arange(182, 196)
    remove_3 = np.arange(14, 169, 14)
    remove_4 = np.arange(27, 182, 14)
    to_remove = np.concatenate([remove_1, remove_2, remove_3, remove_4])
    valid_patches = np.setdiff1d(valid_patches, to_remove)
    return valid_patches

def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def patchify(imgs: torch.Tensor, patch_size: int = 16, image_channels: int = 1) -> torch.Tensor:
    """
    Used to create targets
    imgs: (N, C, H, W)
    x: (N, L, patch_size**2*C)
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], image_channels, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h*w, p, p))
    return x   

def unpatchify(x: torch.Tensor, patch_size: int = 16, image_channels: int = 1) -> torch.Tensor:
    p = patch_size
    h = w = int(x.shape[1]**0.5)
    assert h * w == x.shape[1]   
    x = x.reshape(shape=(x.shape[0], h, w, p, p, image_channels))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], image_channels, h*p, h*p))
    return imgs  

def average_embeddings(embeddings: torch.Tensor, patch_idx: int,num_patches: int = 9, embed_dim: int = 384) -> torch.Tensor:
    temp_embed = torch.zeros(num_patches, embed_dim)
    temp_embed[0, :] = embeddings[0, patch_idx - 15, :]
    temp_embed[1, :] = embeddings[0, patch_idx - 14, :]
    temp_embed[2, :] = embeddings[0, patch_idx - 13, :]
    temp_embed[3, :] = embeddings[0, patch_idx - 1, :]
    temp_embed[4, :] = embeddings[0, patch_idx, :]
    temp_embed[5, :] = embeddings[0, patch_idx + 1, :]
    temp_embed[6, :] = embeddings[0, patch_idx + 13, :]
    temp_embed[7, :] = embeddings[0, patch_idx + 14, :]
    temp_embed[8, :] = embeddings[0, patch_idx + 15, :]
    temp_embed = torch.mean(temp_embed, dim=0).unsqueeze(0)
    return temp_embed

def get_crop_from_patchified_tensor(patchified_tensor: torch.Tensor, patch_idx: int, patch_size: int = 16, image_channels: int = 1) -> torch.Tensor:
    temp = torch.zeros(9, patch_size, patch_size)
    temp[0, :, :] = patchified_tensor[0, patch_idx - 15, :, :]
    temp[1, :, :] = patchified_tensor[0, patch_idx - 14, :, :]
    temp[2, :, :] = patchified_tensor[0, patch_idx - 13, :, :]
    temp[3, :, :] = patchified_tensor[0, patch_idx - 1, :, :]
    temp[4, :, :] = patchified_tensor[0, patch_idx, :, :]
    temp[5, :, :] = patchified_tensor[0, patch_idx + 1, :, :]
    temp[6, :, :] = patchified_tensor[0, patch_idx + 13, :, :]
    temp[7, :, :] = patchified_tensor[0, patch_idx + 14, :, :]
    temp[8, :, :] = patchified_tensor[0, patch_idx + 15, :, :]
    temp = unpatchify(temp.unsqueeze(0), patch_size=patch_size, image_channels=image_channels).squeeze().cpu().numpy()
    return temp

def main():
    set_seeds(args.seed)
    VALID_PATCHES = get_valid_patches() 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_channels = 3 if "imagenet" in args.weights.lower() else 1

    # Load model 
    model, cfg = get_pretrained_model_v2(
        name=args.model,
        weights=args.weights,
        path=None,
        mask_ratio=0.0,
        pretrained=True if n_channels==3 else False,
        in_channels=n_channels,
        as_classifier=True,
        blocks="all",
        global_pool="patch",
        num_classes=4
    )
    model.to(DEVICE)
    model.eval()

    # Load dataset
    _, _, test_dataset = get_dataset(name=args.dataset, cfg=cfg, validation=True)

    key_embeds, key_imgs, key_patches = [], [], [] 
    with torch.no_grad():
        for idx in trange(len(test_dataset)):
            batch = test_dataset[idx]
            X, y = batch 
            X = X.unsqueeze(0).to(DEVICE)
            mask = y.unsqueeze(0)[:, [args.class_id], :, :].to(DEVICE)
            pred = model.forward_features(X)
            if "imagenet" in args.weights.lower():
                X = X[:, [0], :, :]
            patchified_img = patchify(X, patch_size=args.patch_size)
            patchified_mask = patchify(mask, patch_size=args.patch_size)
            for k in VALID_PATCHES:
                temp_embed = pred[0, k, :]
                if idx == args.image_id and k == args.patch_id:
                    query_embed = temp_embed.clone() 
                    temp_img = get_crop_from_patchified_tensor(patchified_img, patch_idx=k, patch_size=args.patch_size)
                    temp_mask = get_crop_from_patchified_tensor(patchified_mask, patch_idx=k, patch_size=args.patch_size)
                    if not os.path.exists("./patch-retrieval-experiment/templates/template.png"):
                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        ax.imshow(temp_img, cmap="hot")
                        ax.axis("off")
                        fig.savefig("./patch-retrieval-experiment/templates/template.png", bbox_inches="tight", dpi=1200)
                        plt.close(fig)
                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        ax.imshow(temp_mask, cmap="gray")
                        ax.axis("off")
                        fig.savefig("./patch-retrieval-experiment/templates/template_mask.png", bbox_inches="tight", dpi=1200)
                        plt.close(fig)
                elif idx == args.image_id and k in [args.patch_id - 15, args.patch_id - 14, args.patch_id - 13, args.patch_id - 1, args.patch_id, args.patch_id + 1, args.patch_id + 13, args.patch_id + 14, args.patch_id + 15]:
                    continue 
                else:
                    key_embeds.append(temp_embed)
                    key_imgs.append(idx)
                    key_patches.append(k)

    key_embeddings = torch.stack(key_embeds, dim=0)
    query_embed = query_embed.squeeze()
    similarities = F.cosine_similarity(query_embed, key_embeddings, dim=-1)
    sorted_similarities, sorted_indices = torch.sort(similarities, descending=True)
    key_imgs = [key_imgs[i] for i in sorted_indices]
    key_patches = [key_patches[i] for i in sorted_indices]
    
    for rank, (sim, idx, patch) in enumerate(zip(sorted_similarities[:args.num_retrievals], key_imgs[:args.num_retrievals], key_patches[:args.num_retrievals])):
        img = test_dataset[idx][0]
        y = test_dataset[idx][1]
        mask = y.unsqueeze(0)[:, [args.class_id], :, :].to(DEVICE)
        if "imagenet" in args.weights.lower():
            img = img[[0], :, :]

        patchified_img = patchify(img.unsqueeze(0), patch_size=args.patch_size)
        patchified_mask = patchify(mask, patch_size=args.patch_size)
        temp_img = get_crop_from_patchified_tensor(patchified_img, patch_idx=patch, patch_size=args.patch_size)
        temp_mask = get_crop_from_patchified_tensor(patchified_mask, patch_idx=patch, patch_size=args.patch_size)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(temp_img, cmap="hot")
        ax.axis("off")
        fig.savefig(f"./patch-retrieval-experiment/candidates/retrieval-{rank}-{args.weights}.png", bbox_inches="tight", dpi=1200)
        plt.close(fig)
        print(f"--- Saved retrieval {rank} of {args.num_retrievals} ---")
    
    

if __name__ == "__main__":
    main()