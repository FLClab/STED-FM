import numpy as np
import os
import glob 
import json
import argparse
import sys
import matplotlib.pyplot as plt
from typing import Dict
from matplotlib import pyplot, patches
sys.path.insert(0, "../../")
from DEFAULTS import BASE_PATH, COLORS
from utils import savefig
from stats import resampling_stats, plot_p_values

parser = argparse.ArgumentParser()
parser.add_argument("--metric", type=str, default="acc")
parser.add_argument("--model", type=str, default="mae-small")
args = parser.parse_args()

print(args)

def load_file(file):
    with open(file, "r") as handle:
        data = json.load(handle)
    return data

def get_data(pretraining="STED"):
    data = {}
    for dataset in ["optim", "neural-activity-states", "peroxisome", "polymer-rings", "dl-sim"]:
        files = glob.glob(os.path.join(BASE_PATH, "baselines", f"{args.model}_{pretraining}", dataset, f"accuracy_linear-probe_None_*.json"), recursive=True)
        if len(files) < 1: 
            print(f"Could not find files for dataset: `{dataset}` and pretraining: `{pretraining}`")
            return data
        if len(files) != 5:
            print(f"Could not find all files for dataset: `{dataset}` and pretraining: `{pretraining}`")
        scores = []
        for file in files:
            scores.append(load_file(file))
        data[dataset] = scores
    return data

def plot_data(sted_data: Dict[str, list], hybrid_data: Dict[str, list]) -> None:
    all_data = []
    for dataset in ["optim", "neural-activity-states", "peroxisome", "polymer-rings", "dl-sim"]:
        sted_data = sted_data[dataset]
        hybrid_data = hybrid_data[dataset]
        diff_data = [hybrid_data[i] - sted_data[i] for i in range(len(sted_data))]
        all_data.append(diff_data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(all_data)
    ax.set_xticklabels(["optim", "neural-activity-states", "peroxisome", "polymer-rings", "dl-sim"])
    ax.set_ylabel("(Hybrid - STED) $\delta$ accuracy")
    fig.savefig("./hybrid.png")
    plt.close(fig)

    
def main():
    mode = "linear-probe"
    sted_data = get_data(pretraining="STED")
    hybrid_data = get_data(pretraining="Hybrid")
    print(sted_data)
    

    plot_data(sted_data=sted_data, hybrid_data=hybrid_data)

if __name__ == "__main__":
    main()