
import numpy
import os, glob
import json
import argparse
import sys
from matplotlib.lines import Line2D
from matplotlib import pyplot

sys.path.insert(0, "../../")
from DEFAULTS import BASE_PATH, COLORS
from utils import savefig

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, 
                    help="Name of the model")
parser.add_argument("--dataset", type=str, 
                    help="Name of the dataset")
parser.add_argument("--best-model", type=str, default=None, 
                    help="Which model to keep")
parser.add_argument("--metric", default="aupr", type=str,
                    help="Name of the metric to access from the saved file")
args = parser.parse_args()

print(args)

def load_file(file):
    with open(file, "r") as handle:
        data = json.load(handle)
    return data

def get_data(mode="from-scratch", pretraining="STED"):
    data = {}
    if mode == "from-scratch":
        files = glob.glob(os.path.join(BASE_PATH, "segmentation-baselines", f"{args.model}", args.dataset, f"{mode}*", f"segmentation-scores.json"), recursive=True)
    else:
        files = glob.glob(os.path.join(BASE_PATH, "segmentation-baselines", f"{args.model}", args.dataset, f"{mode}*_{pretraining.upper()}*", f"segmentation-scores.json"), recursive=True)
        files = [f for f in files if "labels" not in f]

    if mode == "pretrained":
        # remove files that contains samples
        files = list(filter(lambda x: "frozen" not in x, files))   
        
    # remove files that contains samples
    files = list(filter(lambda x: "samples" not in x, files))
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

    averaged = []
    for key, values in data.items():
        values = numpy.array([value[args.metric] for value in values])
        values_masked = numpy.ma.masked_equal(values, -1)

        mean, std = numpy.ma.mean(values_masked, axis=1), numpy.ma.std(values_masked, axis=1)
        mean = numpy.mean(mean, axis=-1)
        averaged.append(mean)
        # ax.bar(position, mean, yerr=std, color=COLORS[pretraining], align="edge", **kwargs)
        ax.scatter([position] * len(mean), mean, color=COLORS[pretraining])
        bplot = ax.boxplot(mean, positions=[position], showfliers=True, **kwargs)
        for name, parts in bplot.items():
            for part in parts:
                part.set_color(COLORS[pretraining])

    return (fig, ax)

def main():
    fig, ax = pyplot.subplots()
    modes = ["pretrained-frozen", "pretrained"]
    pretrainings = ["STED", "SIM", "HPA", "JUMP","ImageNet"]

    width = 1/(len(pretrainings) + 1)
    for j, mode in enumerate(modes):
        for i, pretraining in enumerate(pretrainings):
            data = get_data(mode=mode, pretraining=pretraining)
            
            position = j + i / (len(pretrainings) + 1)
            if mode == "from-scratch":
                position = j + (len(pretrainings) - 1) * width / 2
                fig, ax = plot_data("from-scratch", data, figax=(fig, ax), position=position, widths=0.9*width)
                break
            fig, ax = plot_data(pretraining, data, figax=(fig, ax), position=position, widths=0.9*width)

    ax.set(
        ylabel=args.metric,
        # ylim=(0, 1),
        xticks=numpy.arange(len(modes)) + width * len(pretrainings) / 2 - 0.5 * width,
    )
    ax.set_xticklabels(modes, rotation=30)
    legend_elements = [Line2D([0], [0], color=COLORS[pretraining], label=pretraining, markerfacecolor=COLORS[pretraining]) for pretraining in pretrainings]

    pyplot.legend(handles=legend_elements)

    savefig(fig, os.path.join(".", "results", f"{args.model}_{args.dataset}_scratch-pretrained"), extension="pdf")

if __name__ == "__main__":
    main()