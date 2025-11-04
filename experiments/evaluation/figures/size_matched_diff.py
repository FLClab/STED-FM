import numpy as np 
import matplotlib.pyplot as plt 
import argparse 
import os 
import glob 
from matplotlib import patches 
from stedfm.DEFAULTS import BASE_PATH, COLORS 
from stedfm.utils import savefig 
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mae-small")
parser.add_argument("--mode", type=str, default="linear-probe")
parser.add_argument("--metric", type=str, default="acc")
args = parser.parse_args()

def load_file(file):
    with open(file, "r") as handle:
        data = json.load(handle)
    return data

def get_data(pretraining: str, downstream: str, mode: str):
    full_files = glob.glob(os.path.join(BASE_PATH, "baselines", f"{args.model}_{pretraining}", downstream, f"accuracy_{mode}_None_*.json"), recursive=True)
    matched_files = glob.glob(os.path.join(BASE_PATH, "baselines", f"{args.model}_{pretraining}", downstream, f"size-matched_accuracy_{mode}_None_*.json"), recursive=True)
    if len(full_files) < 1:
        print(f"Could not find files for mode: `{mode}` and pretraining: `{pretraining}`")
        return None, None
    if len(matched_files) < 1:
        print(f"Could not find files for mode: `{mode}` and pretraining: `{pretraining}`")
        return None, None
    if len(full_files) != 5:
        print(f"Could not find all files for mode: `{mode}` and pretraining: `{pretraining}`")
        return None, None
    if len(matched_files) != 5:
        print(f"Could not find all files for mode: `{mode}` and pretraining: `{pretraining}`")
        return None, None
    scores_full = [load_file(file) for file in full_files]
    scores_matched = [load_file(file) for file in matched_files]
    scores_full = [score[args.metric] for score in scores_full]
    scores_matched = [score[args.metric] for score in scores_matched]
    return scores_full, scores_matched

def main():
    MARKERS = ["o", "s", "*", "^", "P"]
    os.makedirs(os.path.join(".", "results"), exist_ok=True)
    pretraining_datasets = ["JUMP"]# , "HPA"]
    downstream_datasets = ["optim", "neural-activity-states", "peroxisome", "polymer-rings", "dl-sim"]

    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)

    for i, pretraining in enumerate(pretraining_datasets):
        for j, downstream in enumerate(downstream_datasets):
            scores_full, scores_matched = get_data(pretraining=pretraining, downstream=downstream, mode=args.mode)
            if pretraining == "JUMP":
                x = [0.25, 0.50]
            else:
                x = [0.75, 1.00]

            scores_full = np.mean(scores_full)
            scores_matched = np.mean(scores_matched)
            ax.plot(x, [scores_full, scores_matched], marker=MARKERS[j], color="dimgrey", ms=10, alpha=0.7, 
                    markerfacecolor="dimgrey", markeredgecolor="dimgrey")
    ax.set_xticks([0.25, 0.50, 0.75, 1.00])
    ax.set_xticklabels(["JUMP-full", "JUMP-976k", "HPA-full", "HPA-976k"])
    ax.set_ylabel(args.metric)
    ax.set_ylim(0.28, 1.02)
    ax.set_xlim(0.20, 1.05)
    fig.savefig(os.path.join(".", "results", f"size-matched_{args.model}_{args.mode}_diff.pdf"), dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    main()