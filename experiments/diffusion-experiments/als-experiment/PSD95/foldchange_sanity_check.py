import numpy as np 
import matplotlib.pyplot as plt 
import torch 
from torch import nn
from tqdm import tqdm, trange 
import random 
import os 
import glob 
from skimage import measure 
from typing import Union, List 
import pickle 
import tifffile
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", type=str, default="")
parser.add_argument("--results-path", type=str, default="/home/frederic/flc-dataset/experiments/diffusion-experiments/als-experiment/PSD95-DIV9/results/MAE_SMALL_STED_als_all_toyoung_RESULTS.npz")
args = parser.parse_args()

def load_raw_images(path: str) -> np.ndarray:
    pass

def load_results(path: str) -> np.ndarray:
    data = np.load(path)
    for k in data.keys():
        print(k)
        print(data[k].shape)
        print("\n")
    return data

def plot_results() -> None:
    features = np.load(f"/home/frederic/flc-dataset/experiments/diffusion-experiments/lerp-results/wavelet_features/MAE_SMALL_STED_activity_all_to{args.direction}_RESULTS.npz")
    num_proteins = np.load(f"/home/frederic/flc-dataset/experiments/diffusion-experiments/lerp-results/wavelet_features/MAE_SMALL_STED_activity_all_to{args.direction}_NUM_PROTEINS.npz")
    feature_names = ["area", "perimeter", "mean_intensity", "eccentricity", "solidity", "1nn_dist"]
    keys = list(features.keys())
    train_features = np.load(f"./{args.boundary}-experiment/{args.channel}/features/train-features.npz")
    block_features, mg_features = train_features["block_features"], train_features["mg_features"]
    # block_features, mg_features = np.array(block_features), np.array(mg_features)   
    for i, f in enumerate(feature_names):
        data = [features[k][:, i] for k in keys] 
        block_data = block_features[:, i]
        mg_data = mg_features[:, i] 
        data.insert(0, block_data)
        data.append( mg_data)
        
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        parts = ax.boxplot(data, medianprops={'color': 'black'}, showfliers=False, patch_artist=True)
        N = len(data)
        for i, pc in enumerate(parts['boxes']):
            color = "grey"
            if i == 0:
                color = "fuchsia" 
            if i == N - 1:
                color = "dodgerblue"
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8], ["Block", "0", "1", "2", "3", "4", "5", "0MgGlyBic"])
        fig.savefig(f"./{args.boundary}-experiment/{args.channel}/results/{args.weights}-{f}-with-train.pdf", dpi=1200, bbox_inches='tight')
        plt.close(fig)

    data = [num_proteins[k] for k in keys]
    block_data = block_features[:, -1]
    mg_data = mg_features[:, -1]
    data.insert(0, block_data)
    data.append(mg_data) 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    parts = ax.boxplot(data, medianprops={'color': 'black'}, showfliers=False, patch_artist=True)
    for i, pc in enumerate(parts['boxes']):
        color = "grey"
        if i == 0:
            color = "fuchsia" 
        if i == N - 1:
            color = "dodgerblue"
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8], ["Block", "0", "1", "2", "3", "4", "5", "0MgGlyBic"])
    fig.savefig(f"./{args.boundary}-experiment/{args.channel}/results/num_proteins-with-train.pdf", dpi=1200, bbox_inches='tight')
    plt.close(fig)

def main():
    results = load_results(path=args.results_path)

if __name__ == "__main__":
    main()
