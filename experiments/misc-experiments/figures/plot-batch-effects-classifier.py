
import pandas
import json
import os
import matplotlib
import argparse
import glob
import numpy

from matplotlib import pyplot

import sys
sys.path.insert(0, "../../")
from DEFAULTS import BASE_PATH
from utils import savefig

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--model", type=str, default='mae-lightning-small')
parser.add_argument("--opts", nargs="+", default=[], 
                    help="Additional configuration options")    
args = parser.parse_args()

# Assert args.opts is a multiple of 2
if len(args.opts) == 1:
    args.opts = args.opts[0].split(" ")
assert len(args.opts) % 2 == 0, "opts must be a multiple of 2"

COLORS = {
    "STED" : "tab:blue",
    "HPA" : "tab:orange",
    "ImageNet" : "tab:red",
    "JUMP" : "tab:green"
}

def load_data(files, batch_effect):
    """
    Load the data from the files

    :params files: dict of files
    :params batch_effect: the name of the batch effect

    :returns: A ``pandas.DataFrame`` with the data
    """
    data = {}
    for key, file in files.items():
        with open(file, "r") as f:
            data[key] = json.load(f)[batch_effect]
    return data

def get_files():
    files = glob.glob(os.path.join("..", "results", "batch-effects-classifier", f"{args.dataset}_{args.model}_*.json"))
    if len(files) == 0:
        raise FileNotFoundError(f"No files found for {args.dataset}_{args.model}_*.json")

    out = {}
    for file in files:
        for key in ["ImageNet", "HPA", "JUMP", "STED"]:
            if key.lower() in file.lower():
                out[key] = file
                break
    return out

def plot(batch_effect, data):
    """
    Plot the data

    :params batch_effect: The name of the batch effect
    :params data: The data
    """
    fig, ax = pyplot.subplots(1, 1, figsize=(3, 3))
    for key, scores in data.items():
        to_plot = []
        for augmentation, values in scores["augmentations"].items():
            to_plot.append(values["accuracy"])
        to_plot = numpy.array(to_plot)

        normalized = to_plot #/ to_plot[0]
        ax.plot(normalized, label=key, color=COLORS[key])

    ax.set(
        ylabel="Accuracy",
        ylim=(0.0, 1.0),
        xticks=numpy.arange(len(scores["augmentations"].keys())),
    )
    ax.set_xticklabels(scores["augmentations"].keys(), rotation=45, ha="right")
    ax.legend()
    savefig(fig, os.path.join("batch-effects-classifier", f"{args.dataset}_{args.model}_{batch_effect}"), save_white=True)

def main():
    effects = [
        "geometric", "poisson", "gaussian-noise", "gaussian-blur", "mixed"
    ]

    files = get_files()
    for effect in effects:
        data = load_data(files, effect)
        plot(effect, data)

if __name__ == "__main__":
    main()