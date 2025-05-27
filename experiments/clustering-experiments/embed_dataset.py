import torch 
import os 
import argparse 
from torch.utils.data import DataLoader 
from tqdm import tqdm 
import numpy as np 
from typing import List, Optional, Tuple 
import sys 
from stedfm.DEFAULTS import BASE_PATH 
from stedfm.loaders import get_dataset 
from stedfm.model_builder import get_pretrained_model_v2 

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="optim")
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--global-pool", type=str, default="avg")
args = parser.parse_args()


if __name__=="__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {DEVICE} ---")

    model, cfg = get_pretrained_model_v2(
        name=args.model,
        weights=args.weights,
        path=None,
        mask_ratio=0.0,
        pretrained=True if "imagenet" in args.weights.lower() else False,
        in_channels=3 if "imagenet" in args.weights.lower() else 1,
        as_classifier=True,
        blocks="all",
        num_classes=4,
    )
    model = model.to(DEVICE)
    model.eval()

    _, _, test_loader = get_dataset(
        name=args.dataset,
        transform=None,
        training=True, 
        path=None,
        batch_size=cfg.batch_size,
        n_channels=3 if "imagenet" in args.weights.lower() else 1,
    )

    embeddings, labels = [], []
    with torch.no_grad():
        for img, metadata in tqdm(test_loader):
            img = img.to(DEVICE)
            label = metadata["label"]
            output = model.forward_features(img)
            print(output.shape)
            embeddings.extend(output.data.cpu().numpy())
            labels.extend(label)

    embeddings = np.array(embeddings)
    labels = np.array(labels)

