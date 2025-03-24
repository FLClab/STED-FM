import matplotlib.pyplot as plt 
import pandas 
from math import pi 
import numpy as np 
import glob 
import argparse 
import sys 
import os 
import json 
from matplotlib import patches 
sys.path.insert(0, "../../")
from DEFAULTS import BASE_PATH, COLORS
from utils import savefig 

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mae-small")
parser.add_argument("--mode", type=str, default="linear-probe")
parser.add_argument("--metric", type=str, default="acc")
args = parser.parse_args() 


def load_file(file):
    with open(file, "r") as handle:
        data = json.load(handle)
    return data

def get_data(pretraining: str, downstream: str, mode: str) -> dict:
    files = glob.glob(os.path.join(BASE_PATH, "baselines", f"{args.model}_{pretraining}", downstream, f"accuracy_{mode}_None_*.json"), recursive=True)
    if len(files) < 1: 
        print(f"Could not find files for mode: `{mode}` and pretraining: `{pretraining}`")
        return data
    if len(files) != 5:
        print(f"Could not find all files for mode: `{mode}` and pretraining: `{pretraining}`")
    scores = []
    for file in files:
        scores.append(load_file(file))
    scores = [value[args.metric] for value in scores]
    return scores 


def main():
    pretraining_datasets = ["STED", "SIM", "HPA", "JUMP", "ImageNet"] if args.model == "mae-small" else ["STED"] 
    downstream_datasets =  ["optim", "neural-activity-states", "peroxisome", "polymer-rings", "dl-sim"]
    N = len(downstream_datasets)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1], downstream_datasets)
    ax.set_rlabel_position(0)
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax.set_ylim(0.7, 1.0)

    max_values = [0.0, 0.0, 0.0, 0.0, 0.0]
    for i, pretraining in enumerate(pretraining_datasets):
        values = []
        for j, downstream in enumerate(downstream_datasets):
            scores = get_data(pretraining=pretraining, downstream=downstream, mode=args.mode)
            acc = np.mean(scores)
            if acc > max_values[j]:
                max_values[j] = acc
            values.append(acc)
        values = [maxv - v for v, maxv in zip(values, max_values)]
        values = [1.0 - v for v in values]
        values += values[:1]
        ax.plot(angles, values, color=COLORS[pretraining], label=pretraining)
        # ax.fill(angles, values, color=COLORS[i], alpha=0.1)

    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    savefig(fig, os.path.join(".", "results", f"{args.model}_{args.mode}_radar_full_data"), extension="pdf")


    
if __name__=="__main__":
    main()
        