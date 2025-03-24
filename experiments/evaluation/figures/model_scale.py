import numpy as np 
import os 
import json 
import glob 
import argparse 
import sys 
import matplotlib.pyplot as plt 
from matplotlib import patches 
sys.path.insert(0, "../../")
from DEFAULTS import BASE_PATH, COLORS 
from utils import savefig 

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="linear-probe")
parser.add_argument("--metric", type=str, default="acc")
args = parser.parse_args() 

def load_file(file):
    with open(file, "r") as handle:
        data = json.load(handle)
    return data

def get_data(model: str, downstream: str, mode: str) -> dict:
    files = glob.glob(os.path.join(BASE_PATH, "baselines", f"{model}_STED", downstream, f"accuracy_{mode}_None_*.json"), recursive=True)
    if len(files) < 1: 
        print(f"Could not find files for mode: `{mode}` and pretraining: `STED`")
        return data
    if len(files) != 5:
        print(f"Could not find all files for mode: `{mode}` and pretraining: `STED` ({len(files)}/5)")
    scores = []
    for file in files:
        scores.append(load_file(file))
    scores = [value[args.metric] for value in scores]
    return scores


def main():
    models = ["mae-tiny", "mae-small", "mae-base", "mae-large"][::-1]
    downstream_datasets = ["optim", "neural-activity-states", "peroxisome", "polymer-rings", "dl-sim"]
    M, D = len(models), len(downstream_datasets)

    performance_heatmap = np.zeros((M, D))
    for m, model in enumerate(models):
        for d, downstream in enumerate(downstream_datasets):
            scores = get_data(model=model, downstream=downstream, mode=args.mode)
            performance_heatmap[m, d] = np.mean(scores)

    normalized_heatmap = performance_heatmap.copy()
    for col in range(D):
        diff = 1.0 - np.max(performance_heatmap[:, col])
        normalized_heatmap[:, col] += diff

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(normalized_heatmap, cmap="RdPu")

    for i in range(M):
        for j in range(D):
            text = f'{performance_heatmap[i, j]:.3f}'
            color = "black" if normalized_heatmap[i, j] < 0.95 else "white"
            ax.text(j, i, text, ha='center', va='center', color=color)
    
    ax.set_xticks(np.arange(D))
    ax.set_yticks(np.arange(M))
    ax.set_xticklabels(downstream_datasets)
    ax.set_yticklabels(models)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    savefig(fig, os.path.join(".", "results", f"performance_vs_scale_{args.mode}"), extension="pdf")




if __name__=="__main__":
    main()