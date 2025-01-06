import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import random 
import json 
from tqdm import tqdm 
import argparse 
from attribute_datasets import OptimQualityDataset
import os
from torch.utils.data import DataLoader
import sys 
sys.path.insert(0, "../")
from model_builder import get_pretrained_model_v2

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--blocks", type=str, default="all")
parser.add_argument("--global-pool", type=str, default="avg")
parser.add_argument("--num-per-class", type=int, default=None)
parser.add_argument("--precomputed", action="store_true")
parser.add_argument("--split", type=str, default="train")
args = parser.parse_args()

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    model, cfg = get_pretrained_model_v2(
        name=args.model,
        weights=args.weights,
        path=None,
        mask_ratio=0.0,
        pretrained=False,
        in_channels=3 if "imagenet" in args.weights.lower() else 1,
        as_classifier=True, 
        blocks=args.blocks,
        num_classes=2
    )
    model = model.to(device)

    dataset = OptimQualityDataset(
        data_folder=f"/home-local/Frederic/Datasets/evaluation-data/optim_{args.split}",
        num_samples={"actin": None, "tubulin": None, "CaMKII_Neuron": None, "PSD95_Neuron": None},
        apply_filter=True,
        classes=['actin', 'tubulin', 'CaMKII_Neuron', 'PSD95_Neuron'],
        high_score_threshold=0.70,
        low_score_threshold=0.60,
        n_channels=3 if "imagenet" in args.weights.lower() else 1
    )

    print(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False, num_workers=1)

    embeddings = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, data_dict = batch
            images = images.to(device)
            labels = data_dict["label"]
            features = model.forward_features(images)
            embeddings.append(features.cpu().detach().numpy())
            all_labels.extend(labels.cpu().detach().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    print(embeddings.shape)
    all_labels = np.array(all_labels)
    print(all_labels.shape)

    np.savez(f"./lerp-results/embeddings/quality/{args.weights}-optimquality-embeddings_{args.split}.npz", embeddings=embeddings, labels=all_labels)

