import numpy as np 
import os 
import glob 
from tqdm import tqdm 
import argparse 
import sys 
import pandas 
import json
import matplotlib.pyplot as plt
import matplotlib 
sys.path.insert(0, "../../")
from DEFAULTS import BASE_PATH, COLORS
from utils import savefig 

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--metric", type=str, default="aupr")
parser.add_argument("--mode", type=str, default="pretrained-frozen")
args = parser.parse_args()

def load_file(file):
    with open(file, "r") as handle:
        data = json.load(handle)
    return data 

def get_data(pretraining: str, downstream: str):
    files = glob.glob(os.path.join(BASE_PATH, "segmentation-baselines", f"{args.model}", downstream, f"{args.mode}*_{pretraining.upper()}*", f"segmentation-scores.json"), recursive=True)
    files = [f for f in files if "labels" not in f]
    files = [f for f in files if "samples" not in f]
    if args.mode == "pretrained-frozen":
        files = [f for f in files if "frozen" in f]
    if len(files) < 1:
        print(f"Could not find files for pretraining: `{pretraining}`, downstream: `{downstream}`")
        return data
    if len(files) != 5:
        print(f"Could not find all files for pretraining: `{pretraining}`, downstream: `{downstream}`")
    scores = [] 
    for file in files:
        scores.append(load_file(file))
    scores = [value[args.metric] for value in scores]
    return scores

def main():
    pretraining_datasets = ["STED", "SIM", "HPA", "JUMP", "ImageNet"]
    downstream_datasets = ["factin", "synaptic-semantic-segmentation", "footprocess", "lioness"]
    P, D = len(pretraining_datasets), len(downstream_datasets)
    performance_heatmap = np.zeros((P, D))
    for i, pretraining in enumerate(pretraining_datasets):
        for j, downstream in enumerate(downstream_datasets):
            scores = get_data(pretraining=pretraining, downstream=downstream)
            scores_masked = np.ma.masked_equal(scores, -1)
            mean = np.ma.mean(scores_masked, axis=1)
            mean = np.mean(np.mean(mean, axis=-1))
            performance_heatmap[i, j] = mean

    normalized_heatmap = performance_heatmap.copy()
    for col in range(D):
        diff = 1.0 - np.max(performance_heatmap[:, col])
        normalized_heatmap[:, col] += diff

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(normalized_heatmap, cmap="RdPu")
    
    # Add text annotations with performance values
    for i in range(P):
        for j in range(D):
            text = f'{performance_heatmap[i, j]:.3f}'
            color = "black" if normalized_heatmap[i, j] < 0.85 else "white"
            ax.text(j, i, text, ha='center', va='center', color=color)
    
    ax.set_xticks(np.arange(D))
    ax.set_yticks(np.arange(P))
    ax.set_xticklabels(downstream_datasets)
    ax.set_yticklabels(pretraining_datasets)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.colorbar(im)
    savefig(fig, os.path.join(".", "results", f"{args.model}_{args.mode}_full_heatmap"), extension="pdf")
        
            
            

if __name__ == "__main__":
    main()