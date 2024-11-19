
import numpy
import os, glob
import json
import argparse
import sys

from matplotlib import pyplot

sys.path.insert(0, "../../")
from DEFAULTS import BASE_PATH
from utils import savefig

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, 
                    help="Name of the model")
parser.add_argument("--dataset", type=str, 
                    help="Name of the dataset")
parser.add_argument("--best-model", type=str, default=None, 
                    help="Which model to keep")
parser.add_argument("--metric", default="acc", type=str,
                    help="Name of the metric to access from the saved file")
parser.add_argument("--samples", nargs="+", type=str, default=None,
                    help="Number of samples to plot")
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

def get_data(pretraining="STED"):
    data = {}
    for sample in args.samples:
        files = glob.glob(os.path.join(BASE_PATH, "baselines", f"{args.model}_{pretraining}", args.dataset, f"accuracy_linear-probe_{sample}_*.json"), recursive=True)
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

def plot_data(pretraining, data, figax=None):

    if figax is None:
        fig, ax = pyplot.subplots()
    else:
        fig, ax = figax

    averaged = []
    for key, values in data.items():
        values = [value[args.metric] for value in values]

        mean, std = numpy.mean(values), numpy.std(values)
        averaged.append(mean)
        ax.errorbar(float(key), mean, std, color=COLORS[pretraining])

    ax.plot([float(key) for key in data.keys()], averaged, color=COLORS[pretraining], label=pretraining)
    ax.set(
        xlabel="Num. samples per class", ylabel=args.metric,
        ylim=(0, 1),
        xticks=[int(s) for s in data.keys()],
    )

    return (fig, ax)

def main():

    fig, ax = pyplot.subplots(figsize=(4,3))
    for pretraining in ["STED", "HPA", "JUMP", "ImageNet"]:
        data = get_data(pretraining=pretraining)
        fig, ax = plot_data(pretraining, data, figax=(fig, ax))
    ax.legend()
    savefig(fig, os.path.join(".", "results", f"{args.model}_{args.dataset}_small-dataset-samples"), extension="png")

if __name__ == "__main__":
    main()