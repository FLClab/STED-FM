import numpy as np 
import os 
import json 
import argparse 
import glob
import sys 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
sys.path.insert(0, "../../")
from DEFAULTS import BASE_PATH, COLORS
from utils import savefig

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mae-small")
parser.add_argument("--dataset", type=str, default="optim")
parser.add_argument("--metric", type=str, default="acc")
parser.add_argument("--samples", nargs="+", type=str, default=None, help="Number of samples to plot")
parser.add_argument("--mode", type=str, default="finetuned")
args = parser.parse_args()

print(args)

def load_file(file):
    with open(file, "r") as handle:
        data = json.load(handle)
    return data

def get_samples_data(pretraining="STED"):
    data = {}
    for sample in args.samples:
        files = glob.glob(os.path.join(BASE_PATH, "baselines", f"{args.model}_{pretraining}", args.dataset, f"accuracy_{args.mode}_{sample}_*.json"), recursive=True)
        if len(files) < 1:
            print(f"Could not find files for sample: `{sample}` and pretraining: `{pretraining}`")
            continue
        if len(files) != 5:
            print(f"Could not find all files for sample: `{sample}` and pretraining: `{pretraining}`")
        scores = []
        for file in files:
            scores.append(load_file(file))
        data[sample] = scores
    return data

def get_data(mode=args.mode, pretraining="STED"):
    data = {}
    files = glob.glob(os.path.join(BASE_PATH, "baselines", f"{args.model}_{pretraining}", args.dataset, f"accuracy_{mode}_None_*.json"), recursive=True)
    if len(files) < 1: 
        print(f"Could not find files for mode: `{mode}` and pretraining: `{pretraining}`")
        return data
    if len(files) != 5:
        print(f"Could not find all files for mode: `{mode}` and pretraining: `{pretraining}`")
    scores = []
    for file in files:
        scores.append(load_file(file))
    data[mode] = scores
    return data


def main():

    fig = plt.figure()
    ax = fig.add_subplot(111)
    r = np.arange(len(args.samples) + 1).tolist()
    pretrainings = ["STED", "SIM", "HPA", "JUMP", "ImageNet"]
    width = 1/(len(pretrainings)+ 1)
    for i, pretraining in enumerate(pretrainings):
        samples_data = get_samples_data(pretraining=pretraining)
        pretraining_results = []
        for s in args.samples:
            try:
                s_data = [item[args.metric] for item in samples_data[s]]
                pretraining_results.append(s_data)
            except:
                print(f"Could not find data for sample: `{s}` and pretraining: `{pretraining}`")
                pretraining_results.append([])
        full_data = get_data(pretraining=pretraining)
        f_data = [item[args.metric] for item in full_data["finetuned"]]
        pretraining_results.append(f_data)
        parts = ax.boxplot(pretraining_results, positions=r, widths=width)
        for name, parts in parts.items():
            for part in parts:
                part.set_color(COLORS[pretraining])
        r = [item + width for item in r]

    ax.set(
        ylabel=args.metric,
        # ylim=(0, 1),
        xticks=np.arange(len(args.samples) + 1) + width * len(pretrainings) / 2 - 0.5 * width,
        xticklabels=args.samples + ["Full"]
    )
    ax.legend(
        handles=[
            patches.Patch(color=COLORS[label], label=label) for label in pretrainings
        ],
        fontsize=8
    )
    savefig(fig, os.path.join(".", "results", f"boxplot_{args.model}_{args.dataset}_{args.mode}_small_data"), extension="pdf")

        

if __name__ == "__main__":
    main()

