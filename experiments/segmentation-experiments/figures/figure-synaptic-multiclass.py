import numpy as np 
import matplotlib.pyplot as plt 
import json 
import argparse 
import sys 
from matplotlib.lines import Line2D
import glob  
import os 
import matplotlib.pyplot as plt 
sys.path.insert(0, "../../")
from DEFAULTS import BASE_PATH, COLORS 
from utils import savefig 

parser = argparse.ArgumentParser() 
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--metric", type=str, default="aupr")
parser.add_argument("--mode", type=str, default="pretrained")
parser.add_argument("--dataset", type=str, default="synaptic-semantic-segmentation")
args = parser.parse_args()

def load_file(file: str):
    with open(file, "r") as handle:
        data = json.load(handle)
    return data

def get_data(mode="from-scratch", pretraining="STED"):
    data = {}
    if mode == "from-scratch":
        files = glob.glob(os.path.join(BASE_PATH, "segmentation-baselines", f"{args.model}", args.dataset, f"pretrained*", f"segmentation-scores.json"), recursive=True)
    else:
        files = glob.glob(os.path.join(BASE_PATH, "segmentation-baselines", f"{args.model}", args.dataset, f"pretrained*_{pretraining.upper()}*", f"segmentation-scores.json"), recursive=True)
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

def main():
    classes = ["Round", "Elongated", "Perforated", "Multi-domain"]
    pretrainings = ["STED", "SIM", "HPA", "JUMP", "ImageNet"] 
    width = 1/(len(pretrainings) + 1)
    positions = np.arange(len(classes))
    for c in range(len(classes)):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        width = 1/(len(pretrainings) + 1)
        cls_boxes = []
        for i, pretraining in enumerate(pretrainings):
            data = get_data(mode=args.mode, pretraining=pretraining)
            data = data[args.mode]
            values = np.array([value[args.metric] for value in data])
        
            values_masked = np.ma.masked_equal(values, - 1)
            mean, std = np.ma.mean(values_masked, axis=1), np.ma.std(values_masked, axis=1)
            mean = mean[:, c]
            print(f"Pretraining: {pretraining}, Class: {classes[c]}, Mean: {np.mean(mean)}")
            ax.scatter([positions[c] + width * i] * len(mean), mean, color=COLORS[pretraining])
            bplot = ax.boxplot(mean, positions=[positions[c] + width * i], showfliers=True)
            for name, parts in bplot.items():
                for part in parts:
                    part.set_color(COLORS[pretraining])
        ax.set(
            ylabel=args.metric,
            xticklabels=[],
            xticks=[]
        )
        legend_elements = [Line2D([0], [0], color=COLORS[pretraining], label=pretraining, markerfacecolor=COLORS[pretraining]) for pretraining in pretrainings]

        plt.legend(handles=legend_elements)
        savefig(fig, os.path.join(".", "results", f"{args.model}_{classes[c]}_synaptic-semantic-segmentation"), extension="pdf")
            

        # fig, ax = plot_data(pretraining, data, figax=(fig, ax), position=i, widths=width)


if __name__=="__main__":
    main()
    