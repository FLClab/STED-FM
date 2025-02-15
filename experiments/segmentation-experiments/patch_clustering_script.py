import numpy as np 
import matplotlib.pyplot as plt 
import torch 
from torch.utils.data import Dataset, DataLoader 
import argparse 
from typing import Optional, Callable, Tuple 
import io 
from skimage.filters import threshold_otsu 
from tqdm import tqdm, trange 
import torch.nn.functional as F 
import os 
import sys 
from decoders import get_decoder 
from datasets import get_dataset
from sklearn.cluster import KMeans
sys.path.insert(0, "../")
from DEFAULTS import BASE_PATH 
from model_builder import get_pretrained_model_v2 
from utils import update_cfg, save_cfg 
from configuration import Configuration 


# NOTE: mask channels are (rings, fibers, foreground)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="factin")
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--opts", nargs="+", default=[], help="Additional configuration options")
args = parser.parse_args()

def get_n_classes(dataset: str) -> int:
    if dataset in ["factin", "footprocess", "lioness"]:
        return 2 
    elif dataset == "synaptic-semantic-segmentation":
        return 4 
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def cluster_embeddings(embeddings: np.ndarray, n_clusters: int, n_images: int, foreground_masks: np.ndarray) -> np.ndarray:
    print("--- Clustering patch embeddings --- ")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    foreground_indices = np.where(foreground_masks == 1)[0]
    new_data = np.zeros(embeddings.shape[0])
    
    kmeans.fit(embeddings[foreground_indices])
    foreground_data = [item + 1 for item in kmeans.labels_]

    new_data[foreground_indices] = foreground_data 
    opposite_indices = np.setdiff1d(np.arange(embeddings.shape[0]), foreground_indices)
    new_data[opposite_indices] = 0
    labels_per_image = np.split(new_data, n_images)
    return labels_per_image

def display_predictions(preds: np.ndarray, ground_truths: np.ndarray, images: np.ndarray) -> None:
    for i in range(len(preds)):
        pred = preds[i]
        gt = ground_truths[i]
        # gt = np.sum(gt, axis=0)
        img = images[i]
        pred = torch.tensor(pred, dtype=torch.float32)
        pred = pred.view(1, 1, 14, 14) 
        pred = F.interpolate(pred, (224, 224), mode="nearest").squeeze().cpu().int().numpy()
        fig, axs = plt.subplots(1, 4)
        axs[0].imshow(img, cmap="hot")
        axs[1].imshow(gt[0], cmap='gray')
        axs[2].imshow(gt[1], cmap='gray')
        axs[3].imshow(pred, cmap='magma')
        for ax in axs:
            ax.axis("off")
        fig.savefig(f"./temp_{i}.png", dpi=1200)
        plt.close(fig)


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {DEVICE} --- ")
    n_channels = 3 if "imagenet" in args.weights.lower() else 1

    model, cfg = get_pretrained_model_v2(
        name=args.model,
        weights=args.weights,
        path=None,
        mask_ratio=0.0,
        pretrained=True if n_channels == 3 else False,
        in_channels=n_channels,
        as_classifier=True,
        blocks="all",
        global_pool="patch",
        num_classes=4,
    )
    model = model.to(DEVICE)
    model.eval()
    _, _, test_dataset = get_dataset(name=args.dataset, cfg=cfg)


    embeddings = [] 
    ground_truths = []
    images = []
    foreground_masks = []
    with torch.no_grad():
        for i in range(len(test_dataset)):
            batch = test_dataset[i]
            X, y = batch
            X = X.unsqueeze(0).to(DEVICE)
            pred = model.forward_features(X)
            embeddings.append(pred.squeeze(0).cpu().numpy())
            ground_truths.append(y.data.cpu().numpy())
            images.append(X.squeeze(0).cpu().numpy())
            img_numpy = X.squeeze().cpu().numpy()
            ys = np.arange(0, img_numpy.shape[0], 16)
            xs = np.arange(0, img_numpy.shape[1], 16)
            foreground = y.squeeze()[-1].cpu().numpy()
            for y in ys:
                for x in xs:
                    patch = foreground[y:y+16, x:x+16]
                    fg_pixels = np.count_nonzero(patch)
                    total_pixels = patch.shape[0] * patch.shape[1]
                    ratio = fg_pixels / total_pixels
                    if ratio > 0.7:
                        foreground_masks.append(1)
                    else:
                        foreground_masks.append(0)
            if i > 10:
                break
            

    embeddings = np.concatenate(embeddings, axis=0)
    ground_truths = np.array(ground_truths)
    images = np.concatenate(images, axis=0)
    n_images = images.shape[0]
    foreground_masks = np.array(foreground_masks)
    n_classes = get_n_classes(args.dataset)
    preds = cluster_embeddings(embeddings=embeddings, n_clusters=n_classes+1, n_images=n_images, foreground_masks=foreground_masks)
    display_predictions(preds=preds, ground_truths=ground_truths, images=images)


        




if __name__=="__main__":
    main()