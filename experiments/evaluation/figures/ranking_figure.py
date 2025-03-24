import numpy as np 
import matplotlib.pyplot as plt 
import os
import glob 
import json 
import argparse 
from tqdm import tqdm
from matplotlib import patches
import sys 
sys.path.insert(0, "../../")
from DEFAULTS import BASE_PATH, COLORS, MARKERS 
from utils import savefig 

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mae-small")
parser.add_argument("--mode", type=str, default="linear-probe")
parser.add_argument("--metric", type=str, default="acc")
args = parser.parse_args()


def load_file(file):
    with open(file, "r") as handle:
        data = json.load(handle)
    return data 

def get_data(sample_size: str, pretraining: str, downstream: str, mode: str):
    files = glob.glob(os.path.join(BASE_PATH, "baselines", f"{args.model}_{pretraining}", downstream, f"accuracy_{mode}_{sample_size}_*.json"), recursive=True)
    if len(files) < 1:
        print(f"Could not find files for sample: `{sample_size}`, pretraining: `{pretraining}`, downstream: `{downstream}`")
    if len(files) != 5:
        print(f"Could not find all files for sample: `{sample_size}`, pretraining: `{pretraining}`, downstream: `{downstream}` ({len(files)}/5)")
    scores = []
    for file in files:
        scores.append(load_file(file))
    scores = [value[args.metric] for value in scores]
    return scores

def plot_rankings(rankings: dict):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    width = 1 / (len(rankings.keys()) + 1)

    for i, pretraining in enumerate(rankings.keys()):
        sample_scores = []
        for s, sample in enumerate(rankings[pretraining].keys()):
            position = s + i / (len(rankings.keys()) + 1)
            data = rankings[pretraining][sample] 
            err = np.array([0, np.std(data)])[..., np.newaxis]
            bplot = ax.bar(position, np.mean(data), width=width, edgecolor=COLORS[pretraining], facecolor=COLORS[pretraining], yerr=err, alpha=0.4)
            ax.scatter([position] * len(data), data, color=COLORS[pretraining])
            # bplot = ax.boxplot(data, positions=[position], showfliers=True, patch_artist=True)
            # for name, parts in bplot.items():
            #     for part in parts:
            #         if name == 'boxes':
            #             # Set the fill color with alpha
            #             part.set_facecolor(COLORS[pretraining])
            #             part.set_alpha(0.3)
            #             # Set the edge color
            #             # part.set_edgecolor("black")
            #         else:
            #             # For all other elements (whiskers, caps, medians, fliers)
            #             part.set_color(COLORS[pretraining])
            #         part.set_linewidth(1.5)

            
    ax.set(
        ylabel="$acc^* - acc$",
        xticks=np.arange(4) + width * len(rankings.keys()) / 2 - 0.5 * width,
        xticklabels=["10", "25", "50", "100"]
    )
    # ax.legend(
    #     handles=[
    #         patches.Patch(color=COLORS[label], label=label) for label in rankings.keys()
    #     ],
    #     fontsize=8
    # )
    savefig(fig, os.path.join(".", "results", f"ranking_figure_{args.model}_{args.mode}"), extension="pdf")


def main():
    sample_size = ["10", "25", "50", "100"]
    downstream_datasets = ["optim", "neural-activity-states", "peroxisome", "polymer-rings", "dl-sim"]
    pretraining_datasets = ["from-scratch","ImageNet", "JUMP", "HPA", "SIM", "STED"]

    rankings = {
        pretraining: {
            num_samples: [] for num_samples in sample_size
        } for pretraining in pretraining_datasets
    }
    
    for i, sample in tqdm(enumerate(sample_size), total=len(sample_size)):
        for j, downstream in enumerate(downstream_datasets):
            all_scores = []
            for k, pretraining in enumerate(pretraining_datasets):
                scores = get_data(sample_size=sample, pretraining=pretraining, downstream=downstream, mode=args.mode if pretraining != "from-scratch" else "from-scratch")
                all_scores.append(np.mean(scores))

            max_score = max(all_scores)
            delta_scores = [max_score - score for score in all_scores]
            print(f"--- {sample} ; {downstream} ---")
            print(all_scores)
            print(delta_scores)
            print("-----------------\n")
            for p in range(len(pretraining_datasets)):
                rankings[pretraining_datasets[p]][sample].append(delta_scores[p])

    for pretraining in rankings.keys():
        rankings[pretraining]["100"].pop(2) # remove the peroxisome - 100 samples case

    plot_rankings(rankings)

if __name__=="__main__":
    main()