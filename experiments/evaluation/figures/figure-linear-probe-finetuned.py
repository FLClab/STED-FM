
import numpy
import os, glob
import json
import argparse
import sys
import pandas

from matplotlib import pyplot, patches

sys.path.insert(0, "../../")
from DEFAULTS import BASE_PATH
from utils import savefig
from stats import resampling_stats, plot_p_values

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
    "JUMP" : "tab:green",
    "SIM" : "tab:pink",
    "from-scratch" : "silver",
}

def load_file(file):
    with open(file, "r") as handle:
        data = json.load(handle)
    return data

def get_data(mode="linear-probe", pretraining="STED"):
    data = {}
    if mode == "from-scratch":
        files = glob.glob(os.path.join(BASE_PATH, "baselines", f"{args.model}_from-scratch", args.dataset, f"accuracy_{mode}_None_*.json"), recursive=True)
    else:
        files = glob.glob(os.path.join(BASE_PATH, "baselines", f"{pretraining}", args.dataset, f"accuracy_{mode}_None_*.json"), recursive=True)
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

def plot_data(pretraining, data, figax=None, position=0, **kwargs):

    if figax is None:
        fig, ax = pyplot.subplots()
    else:
        fig, ax = figax

    samples = []
    for key, values in data.items():
        values = [value[args.metric] for value in values]

        mean, std = numpy.mean(values), numpy.std(values)
        samples.append(values)
        # ax.bar(position, mean, yerr=std, color=COLORS[pretraining], align="edge", **kwargs)
        ax.scatter([position] * len(values), values, color=COLORS[pretraining])
        bplot = ax.boxplot(values, positions=[position], showfliers=True, **kwargs)
        for name, parts in bplot.items():
            for part in parts:
                part.set_color(COLORS[pretraining])

    return (fig, ax), samples

def main():

    fig, ax = pyplot.subplots(figsize=(4,3))
    modes = ["from-scratch", "linear-probe", "finetuned"]
    pretrainings = ["STED", "SIM", "HPA", "JUMP", "ImageNet"]
    WEIGHTS = {
        "STED" : f"{args.model.upper()}_SIMCLR_STED",
        "SIM" : f"{args.model.upper()}_SIMCLR_SIM",
        "HPA" : f"{args.model.upper()}_SIMCLR_HPA",
        "JUMP" : f"{args.model.upper()}_SIMCLR_JUMP",
        "ImageNet" : f"{args.model.upper()}_IMAGENET1K_V1",
    }

    width = 1/(len(pretrainings) + 1)
    samples = {}
    for j, mode in enumerate(modes):
        for i, pretraining in enumerate(pretrainings):
            data = get_data(mode=mode, pretraining=WEIGHTS[pretraining])
            position = j + i / (len(pretrainings) + 1)
            if mode == "from-scratch":
                position = j + (len(pretrainings) - 1) * width / 2
                (fig, ax), samps = plot_data("from-scratch", data, figax=(fig, ax), position=position, widths=0.9*width)
                samples[f"{mode}"] = samps
                break
            (fig, ax), samps = plot_data(pretraining, data, figax=(fig, ax), position=position, widths=0.9*width)

            samples[f"{mode}-{pretraining}"] = samps
    
    ax.set(
        ylabel=args.metric,
        # ylim=(0, 1),
        xticks=numpy.arange(len(modes)) + width * len(pretrainings) / 2 - 0.5 * width,
        xticklabels=modes
    )
    ax.legend(
        handles=[
            patches.Patch(color=COLORS[label], label=label) for label in pretrainings
        ]
    )
    savefig(fig, os.path.join(".", "results", f"{args.model}_{args.dataset}_linear-probe-finetuned"), extension="png", save_white=True)

    # Calculate statistics
    values = []
    for samps in samples.values():
        values.extend(samps)
    
    p_values, F_p_values = resampling_stats(values, list(samples.keys()))
    print(F_p_values)
    print(p_values)
    if F_p_values < 0.05:
        fig, ax = plot_p_values(p_values)
        savefig(fig, os.path.join(".", "results", f"{args.model}_{args.dataset}_linear-probe-finetuned_stats"), extension="png")


if __name__ == "__main__":
    main()