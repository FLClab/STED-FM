import numpy as np 
import matplotlib.pyplot as plt 
import os
from typing import Tuple
import glob 
import sys 
import json
import argparse
from matplotlib import patches 
sys.path.insert(0, "../../") 
from DEFAULTS import BASE_PATH, COLORS 
from utils import savefig 
from stats import resampling_stats, plot_p_values 

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mae-small")
parser.add_argument("--dataset", type=str, default="optim")
parser.add_argument("--metric", type=str, default="acc")
args = parser.parse_args()

SAMPLES = [10, 25, 50, 100]

PARETO_DATA = []

FLOPS_PER_EPOCH = {
    "optim": {
        "linear-probe": 0.460e9,
        "finetuned": 6446.9e9,
        "linear-probe_samples-10": 0.019e9,
        "linear-probe_samples-25": 0.038e9,
        "linear-probe_samples-50": 0.066e9,
        "linear-probe_samples-100": 0.123e9,
        "finetuned_samples-10": 268.6e9,
        "finetuned_samples-25": 537.2e9,
        "finetuned_samples-50": 940.2e9,
        "finetuned_samples-100": 1746.0e9,
    },
    "neural-activity-states": {
        "linear-probe": 0.693e9,
        "finetuned": 9804.6e9,
        "linear-probe_samples-10": 0.019e9,
        "linear-probe_samples-25": 0.038e9,
        "linear-probe_samples-50": 0.066e9,
        "linear-probe_samples-100": 0.123e9,
        "finetuned_samples-10": 268.6e9,
        "finetuned_samples-25": 537.2e9,
        "finetuned_samples-50": 940.2e9,
        "finetuned_samples-100": 1746.0e9,
    },
    "peroxisome": {
        "linear-probe": 0.114e9,
        "finetuned": 3223.4e9,
        "linear-probe_samples-10": 0.019e9,
        "linear-probe_samples-25": 0.038e9,
        "linear-probe_samples-50": 0.066e9,
        "linear-probe_samples-100": 0.123e9,
        "finetuned_samples-10": 268.6e9,
        "finetuned_samples-25": 537.2e9,
        "finetuned_samples-50": 940.2e9,
        "finetuned_samples-100": 1746.0e9,
    },
    "polymer-rings": {
        "linear-probe": 0.056e9,
        "finetuned": 1611.7e9,
        "linear-probe_samples-10": 0.019e9,
        "linear-probe_samples-25": 0.038e9,
        "linear-probe_samples-50": 0.066e9,
        "linear-probe_samples-100": 0.123e9,
        "finetuned_samples-10": 268.6e9,
        "finetuned_samples-25": 537.2e9,
        "finetuned_samples-50": 940.2e9,
        "finetuned_samples-100": 1746.0e9
    }, 
    "dl-sim": {
        "linear-probe": 1747.4e9,
        "finetuned": 24713.0e9,
        "linear-probe_samples-10": 0.019e9,
        "linear-probe_samples-25": 0.038e9,
        "linear-probe_samples-50": 0.066e9,
        "linear-probe_samples-100": 0.123e9,
        "finetuned_samples-10": 268.6e9,
        "finetuned_samples-25": 537.2e9,
        "finetuned_samples-50": 940.2e9,
        "finetuned_samples-100": 1746.0e9,
    }
    
}

MODEL_MARKERS = {
    "linear-probe": "o",
    "finetuned": "s", 
    "linear-probe_samples-10": "v",
    "linear-probe_samples-25": "<",
    "linear-probe_samples-50": "+",
    "linear-probe_samples-100": "p",
    "finetuned_samples-10": "^",
    "finetuned_samples-25": ">",
    "finetuned_samples-50": "P",
    "finetuned_samples-100": "D",
}


def load_file(file):
    with open(file, "r") as handle:
        data = json.load(handle)
    return data

def get_pareto_front(points):
    """Calculate the Pareto front from a set of 2D points (FLOPS, performance).
    
    Args:
        points: List of [x, y] points where x is FLOPS (to minimize) and y is performance (to maximize)
    Returns:
        List of points that form the Pareto front, sorted by x value
    """
    points_coords = np.array([item[:2] for item in points])
    # Sort points by x value (FLOPS)
    points_sorted = points_coords[points_coords[:, 0].argsort()]
    indices_sorted = np.argsort(points_coords[:, 0])
    
    pareto_front = []
    max_y = float('-inf')
    
    # Since points are sorted by x, we just need to check if y is increasing
    for i, point in enumerate(points_sorted):
        if point[1] > max_y:
            pareto_front.append(points[indices_sorted[i]])
            max_y = point[1]
    
    return pareto_front

def get_data(mode="linear-probe", pretraining="STED"):
    data = {}
    files = glob.glob(os.path.join(BASE_PATH, "baselines", f"{args.model}_{pretraining}", args.dataset, f"accuracy_{mode}_None_*.json"), recursive=True)
    if len(files) < 1: 
        print(f"Could not find files for mode: `{mode}` and pretraining: `{pretraining}`")
        return data
    if len(files) != 5:
        print(f"Could not find all files for mode: `{mode}` and pretraining: `{pretraining}`")
    scores = []
    for f in files:
        scores.append(load_file(file=f))
        data[mode] = scores 
    return data
    

def get_small_data(mode: str ="linear-probe", pretraining: str = "STED"): 
    data = {}
    for sample in SAMPLES:
        files = glob.glob(os.path.join(BASE_PATH, "baselines", f"{args.model}_{pretraining}", args.dataset, f"accuracy_{mode}_{sample}_*.json"), recursive=True)
        if len(files) < 1:
            print(f"Could not find files for mode: `{mode}` and pretraining: `{pretraining}` and sample: `{sample}`")
            continue
        if len(files) != 5:
            print(f"Could not find all files for mode: `{mode}` and pretraining: `{pretraining}` and sample: `{sample}`")
        scores = []
        for f in files:
            scores.append(load_file(file=f))
        data[sample] = scores
    return data

def plot_data(pretraining: str, mode: str, data: dict, figax: Tuple=None, **kwargs):
    global PARETO_DATA
    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax

    if "samples" in mode:
        training_mode = mode.split("_")[0]
        for sample in SAMPLES:
            all_values = data[sample]
            values = [item[args.metric] for item in all_values]
            mean, std = np.mean(values), np.std(values)
            key = f"{training_mode}_samples-{sample}"
            flops = (FLOPS_PER_EPOCH[args.dataset][key] * 300) #/ 1e9
            print(f"Pretraining: {pretraining}, Mode: {mode}, Sample: {sample}, Accuracy: {mean}")
            ax.scatter(flops, mean, marker='o', c=COLORS[pretraining], alpha=0.5)
            PARETO_DATA.append([flops, mean, pretraining, key])
    else:
        all_values = data[mode]
        values = [item[args.metric] for item in all_values]
        mean, std = np.mean(values), np.std(values)
        flops = (FLOPS_PER_EPOCH[args.dataset][mode] * 300) #/ 1e9
        print(f"Pretraining: {pretraining}, Mode: {mode}, Accuracy: {mean}")
        ax.scatter(flops, mean, marker='o', c=COLORS[pretraining], alpha=0.5)
        PARETO_DATA.append([flops, mean, pretraining, mode])
    return (fig, ax)
        

def main():
    global PARETO_DATA
    PARETO_DATA = []
    fig, ax = plt.subplots()
    modes = ["linear-probe", "finetuned", "linear-probe_samples", "finetuned_samples"]
    pretrainings = ["STED", "SIM", "HPA", "JUMP", "ImageNet"] 
    for j, mode in enumerate(modes):
        for i, pretraining in enumerate(pretrainings):
            if "samples" in mode:
                sample_mode = mode.split("_")[0]
                data = get_small_data(mode=sample_mode, pretraining=pretraining)
                (fig, ax) = plot_data(pretraining=pretraining, mode=mode, data=data, figax=(fig, ax))
            else:
                data = get_data(mode=mode, pretraining=pretraining)
                (fig, ax) = plot_data(pretraining=pretraining, mode=mode, data=data, figax=(fig, ax))


    pareto_front = get_pareto_front(PARETO_DATA)
    for p in pareto_front:
        ax.scatter(p[0], p[1], marker='o', c=COLORS[p[2]], alpha=1.0)
    ax.set(
        ylabel=args.metric,
        xlabel="Total FLOPs during training"
    )
    ax.legend(
        handles=[
            patches.Patch(color=COLORS[label], label=label) for label in pretrainings
        ],
        fontsize=8
    )
    ax.set_xscale("log")

    savefig(fig, os.path.join(".", "results", f"{args.model}_{args.dataset}_performance_vs_flops"), extension="pdf")


if __name__ == "__main__":
    main()