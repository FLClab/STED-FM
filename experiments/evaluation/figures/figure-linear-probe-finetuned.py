
import numpy
import os, glob
import json
import argparse
import sys

from matplotlib import pyplot

sys.path.insert(0, "../../")
from DEFAULTS import BASE_PATH

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, 
                    help="Name of the model")
parser.add_argument("--dataset", type=str, 
                    help="Name of the dataset")
parser.add_argument("--best-model", type=str, default=None, 
                    help="Which model to keep")
parser.add_argument("--metric", default="acc", type=str,
                    help="Name of the metric to access from the saved file")
args = parser.parse_args()

print(args)

COLORS = {
    "STED" : "tab:blue",
    "HPA" : "tab:orange",
    "ImageNet" : "tab:red",
    "JUMP" : "tab:green"
}

def load_file(file):
    with open(file, "r") as handle:
        data = json.load(handle)
    return data

def get_data(mode="linear-probe", pretraining="STED"):
    data = {}
    files = glob.glob(os.path.join(BASE_PATH, "baselines", f"{args.model}_{pretraining}", args.dataset, f"accuracy_{mode}_None_*.json"), recursive=True)
    if len(files) < 1: 
        print(f"Could not find files for mode: `{mode}` and pretraining: `{pretraining}`")
        return data
    
    scores = []
    for file in files:
        scores.append(load_file(file))
    data[mode] = scores
    return data

def plot_data(pretraining, data, figax=None, position=0, **kwargs):

    if figax is None:
        fig, ax = pyplot.subplots()
    else:
        fig, ax = figax

    averaged = []
    for key, values in data.items():
        values = [value[args.metric] for value in values]

        mean, std = numpy.mean(values), numpy.std(values)
        averaged.append(mean)
        ax.bar(position, mean, yerr=std, color=COLORS[pretraining], align="edge", **kwargs)

    return (fig, ax)

def main():

    fig, ax = pyplot.subplots(figsize=(4,3))
    modes = ["linear-probe", "finetuned"]
    pretrainings = ["STED", "HPA", "JUMP", "ImageNet"]

    for j, mode in enumerate(modes):
        for i, pretraining in enumerate(pretrainings):
            data = get_data(mode=mode, pretraining=pretraining)
            fig, ax = plot_data(pretraining, data, figax=(fig, ax), position=j + i / (len(pretrainings) + 1), width=1/(len(pretrainings) + 1))

    ax.set(
        ylabel=args.metric,
        ylim=(0, 1),
        xticks=numpy.arange(len(modes)) + 1/(len(pretrainings) + 1) * len(pretrainings) / 2,
        xticklabels=modes
    )

    fig.savefig(os.path.join(".", "results", f"{args.dataset}_linear-probe-finetuned.png"), bbox_inches="tight")

if __name__ == "__main__":
    main()