
import numpy
import os, glob
import json
import argparse
import sys
from scipy import stats
from matplotlib import pyplot

sys.path.insert(0, "../../")
from DEFAULTS import BASE_PATH, COLORS, MARKERS
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
parser.add_argument("--samples", nargs="+", type=str, default=None,
                    help="Number of samples to plot")         
parser.add_argument("--mode", type=str, default="pretrained-frozen", choices=["from-scratch", "pretrained-frozen", "pretrained"],
                    help="Number of samples to plot")                    
args = parser.parse_args()

print(args)


def load_file(file):
    with open(file, "r") as handle:
        data = json.load(handle)
    return data

def get_data(pretraining="STED"):
    data = {}
    for sample in args.samples:
        if args.mode == "from-scratch":
            files = glob.glob(os.path.join(BASE_PATH, "segmentation-baselines", f"{args.model}", args.dataset, f"{args.mode}*-{sample}%-labels*", f"segmentation-scores.json"), recursive=True)
        else:
            files = glob.glob(os.path.join(BASE_PATH, "segmentation-baselines", f"{args.model}", args.dataset, f"{args.mode}*_{pretraining.upper()}*-{sample}%-labels*", f"segmentation-scores.json"), recursive=True)


        if args.mode == "pretrained":
            # remove files that contains samples
            files = list(filter(lambda x: "frozen" not in x, files))  

        # remove files that contains samples
        if len(files) < 1: 
            print(f"Could not find files for mode: `{args.mode}` and pretraining: `{pretraining}`")
            return data
        if len(files) != 5:
            print(f"Could not find all files for mode: `{args.mode}` and pretraining: `{pretraining}` ({len(files)})")
        scores = []
        for file in files:
            scores.append(load_file(file))
        data[sample] = scores
    return data

def get_full_data(mode=args.mode, pretraining="STED"):
    data = {}
    if mode == "from-scratch":
        files = glob.glob(os.path.join(BASE_PATH, "segmentation-baselines", f"{args.model}", args.dataset, f"{mode}*", f"segmentation-scores.json"), recursive=True)
    else:
        files = glob.glob(os.path.join(BASE_PATH, "segmentation-baselines", f"{args.model}", args.dataset, f"{mode}*_{pretraining.upper()}*", f"segmentation-scores.json"), recursive=True)
        files = [f for f in files if "labels" not in f]
        if mode == "pretrained":
            files = [f for f in files if "frozen" not in f]

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

def plot_data(pretraining, data, figax=None, **kwargs):
    full_dataset_results = get_full_data(pretraining=pretraining)
    full_dataset_results = [item[args.metric] for item in full_dataset_results["pretrained"]]
    full_dataset_results = numpy.ma.masked_equal(full_dataset_results, -1)
    full_mean = numpy.mean(full_dataset_results, axis=1)
    full_sem = stats.sem(full_dataset_results, axis=1)
    full_mean = numpy.mean(full_mean)
    full_sem = numpy.mean(full_sem)

    if figax is None:
        fig, ax = pyplot.subplots()
    else:
        fig, ax = figax

    averaged = []
    xs, ys = [], []
    errs = []
    for key, values in data.items():
        values = numpy.array([value[args.metric] for value in values])
        values_masked = numpy.ma.masked_equal(values, -1)

        mean, std = numpy.ma.mean(values_masked, axis=1), numpy.ma.std(values_masked, axis=1)

        sem = stats.sem(values_masked, axis=1)
        errs.append(numpy.mean(sem))

        averaged.append(mean)
        xs.append(float(key))
        ys.append(numpy.mean(mean))

        # ax.errorbar(xs[-1], ys[-1], yerr=numpy.std(mean), color=COLORS[pretraining])

        # ax.bar(position, mean, yerr=std, color=COLORS[pretraining], align="edge", **kwargs)
        # ax.scatter([position] * len(mean), mean, color=COLORS[pretraining])
        # bplot = ax.boxplot(mean, positions=[position], showfliers=True, **kwargs)
        # for name, parts in bplot.items():
        #     for part in parts:
        #         part.set_color(COLORS[pretraining])
    ys.append(full_mean)
    errs.append(full_sem)
    xs.append(100)
    ys = numpy.array(ys)
    errs = numpy.array(errs)
    ax.plot(xs, ys, color=COLORS[pretraining], marker=MARKERS[pretraining])
    ax.fill_between(xs, ys - errs, ys + errs, color=COLORS[pretraining], alpha=0.2)
    return (fig, ax)

def main():

    fig, ax = pyplot.subplots(figsize=(4,3))
    pretrainings = ["STED", "SIM", "HPA", "JUMP", "ImageNet"]
    
    for i, pretraining in enumerate(pretrainings):
        data = get_data(pretraining=pretraining)
        fig, ax = plot_data(pretraining, data, figax=(fig, ax))

    ax.set(
        ylabel=args.metric,
        # ylim=(0, 1),
        xticks=[float(s) for s in args.samples] + [100],
        xticklabels=list(args.samples) + ["Full"]
    )

    savefig(fig, os.path.join(".", "results", f"{args.model}_{args.dataset}_{args.mode}_small-dataset"), extension="pdf")

if __name__ == "__main__":
    main()