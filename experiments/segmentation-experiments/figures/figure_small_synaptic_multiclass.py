import numpy as np 
import matplotlib.pyplot as plt 
import json 
import argparse 
import sys 
from matplotlib.lines import Line2D 
import glob 
import os 
from scipy import stats
sys.path.insert(0, "../../")
from DEFAULTS import BASE_PATH, COLORS, MARKERS 
from utils import savefig 

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--dataset", type=str, default="synaptic-semantic-segmentation")
parser.add_argument("--samples", nargs="+", type=int, default=None)
parser.add_argument("--mode", type=str, default="pretrained")
parser.add_argument("--sampling-mode", type=str, default="labels")
parser.add_argument("--metric", type=str, default="aupr")
args = parser.parse_args()

def load_file(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data 

def get_data(pretraining="STED"):
    data = {}
    for sample in args.samples:
        if args.sampling_mode == "samples":
            files = glob.glob(os.path.join(BASE_PATH, "segmentation-baselines", f"{args.model}", args.dataset, f"{args.mode}*_{pretraining.upper()}*-{sample}-samples*", f"segmentation-scores.json"), recursive=True)
            print(len(files))
            print(os.path.join(BASE_PATH, "segmentation-baselines", f"{args.model}", args.dataset, f"{args.mode}*-{pretraining.upper()}*-{sample}-samples*", f"segmentation-scores.json"))
        else:
            files = glob.glob(os.path.join(BASE_PATH, "segmentation-baselines", f"{args.model}", args.dataset, f"{args.mode}*_{pretraining.upper()}*-{sample}%-labels*", f"segmentation-scores.json"), recursive=True)

        if args.mode == "pretrained":
            files = list(filter(lambda x: "frozen" not in x, files))

        if len(files) < 1:
            print(f"Could not find files for mode: `{args.mode}`, sample `{sample}` and pretraining: `{pretraining}`")
            return data
        if len(files) != 5:
            print(f"Could not find all files for mode: `{args.mode}`, sample `{sample}` and pretraining: `{pretraining}` ({len(files)}/5)")

        scores = []
        for file in files:
            scores.append(load_file(file))
        data[sample] = scores
    return data

def get_full_data(mode: str = args.mode, pretraining: str = "STED"):
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
    files = list(filter(lambda x: "labels" not in x, files))
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

def plot_data(pretraining: str, data: dict, class_idx: int = None, figax: tuple = None, **kwargs):
    full_dataset_results = get_full_data(pretraining=pretraining)
    full_dataset_results = [item[args.metric] for item in full_dataset_results[args.mode]]
    full_dataset_results = np.ma.masked_equal(full_dataset_results, -1)
    full_mean, full_std = np.mean(full_dataset_results, axis=1), np.std(full_dataset_results, axis=1)
    full_sem = stats.sem(full_dataset_results, axis=1)
    if class_idx is None:
        full_mean = np.mean(full_mean, axis=1)
        full_std = np.mean(full_std, axis=1)
        full_sem = np.mean(full_sem, axis=1) 
    else:
        full_mean = full_mean[:, class_idx]
        full_std = full_std[:, class_idx]
        full_sem = full_sem[:, class_idx]

    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax 

    averaged, xs, ys, errs = [], [], [], []
    for key, value in data.items():
        values = np.array([item[args.metric] for item in value])
        values_masked = np.ma.masked_equal(values, -1)
        mean, std = np.ma.mean(values_masked, axis=1), np.ma.std(values_masked, axis=1)
        sem = stats.sem(values_masked, axis=1)
        if class_idx is None:
            mean = np.mean(mean, axis=1)
            std = np.mean(std, axis=1)
            sem = np.mean(sem, axis=1)
        else:
            mean = mean[:, class_idx]
            std = std[:, class_idx]
            sem = sem[:, class_idx]
        averaged.append(mean)
        errs.append(np.mean(sem))
        xs.append(float(key))
        ys.append(np.mean(mean))

    ys.append(np.mean(full_mean))
    errs.append(np.mean(full_sem))
    xs.append(200 if args.sampling_mode == "samples" else 100)
    ys = np.array(ys)
    xs = np.array(xs)
    errs = np.array(errs)
    ax.plot(xs, ys, color=COLORS[pretraining], marker=MARKERS[pretraining])
    ax.fill_between(xs, ys - errs, ys + errs, color=COLORS[pretraining], alpha=0.2)
    return (fig, ax)
        

def main():
    classes = ["Overall", "Round", "Elongated", "Perforated", "Multi-domain"]
    class_indices = [None, 0, 1, 2, 3]
    pretrainings = ["STED", "SIM", "HPA", "JUMP", "ImageNet"]

    for c, c_idx in zip(classes, class_indices):
        print(f"--- Processing class : {c if c_idx is not None else 'Overall'} --- ")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, p in enumerate(pretrainings):
            data = get_data(pretraining=p)
            plot_data(pretraining=p, data=data, class_idx=c_idx, figax=(fig, ax))

        if args.sampling_mode == "labels":
            xticklabels = [str(item) + "%" for item in args.samples] + ["100%"]
        elif args.sampling_mode == "samples":
            xticklabels = list(args.samples) + ["Full"]
        else:
            raise ValueError(f"Invalid sampling mode: {args.sampling_mode}")

        ax.set(
            ylabel=args.metric,
            xticks=[float(s) for s in args.samples] + [200 if args.sampling_mode == "samples" else 100],
            xticklabels=xticklabels
        )

        savefig(fig, os.path.join(".", "results", f"{args.sampling_mode}-{args.model}_{args.dataset}-{c}_{args.mode}_small-dataset"), extension="pdf")


if __name__=="__main__":
    main()

