import numpy as np 
import os 
import glob 
from tqdm import tqdm 
import argparse 
import sys 
import pandas 
import json 
import matplotlib.pyplot as plt 
import matplotlib 
sys.path.insert(0, "../../")
from DEFAULTS import BASE_PATH, COLORS 
from utils import savefig 

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--mode", type=str, default="pretrained-frozen")
parser.add_argument("--metric", type=str, default="aupr")
parser.add_argument("--sampling-mode", type=str, default="labels")
parser.add_argument("--samples", nargs="+", type=str, default=None)
args = parser.parse_args()


colors = ["#5F4690", "#1D6996", "#0F8554", "#EDAD08", "#CC503E"]
color_keys = {
    "ImageNet": 0,
    "JUMP": 1,
    "HPA": 2,
    "SIM": 3,
    "STED": 4,
}

custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom", colors)

def load_file(file):
    with open(file, "r") as handle:
        data = json.load(handle)
    return data 

def get_data(pretraining: str, downstream: str, sample_size: str):
    if args.sampling_mode == "samples":
        files = glob.glob(os.path.join(BASE_PATH, "segmentation-baselines", f"{args.model}", downstream, f"{args.mode}*_{pretraining.upper()}*-{sample_size}-samples*", f"segmentation-scores.json"), recursive=True)
    else:
        files = glob.glob(os.path.join(BASE_PATH, "segmentation-baselines", f"{args.model}", downstream, f"{args.mode}*_{pretraining.upper()}*-{sample_size}%-labels*", f"segmentation-scores.json"), recursive=True)

    if args.mode == "pretrained":
            # remove files that contains samples
            files = list(filter(lambda x: "frozen" not in x, files))  
    if len(files) < 1:
        print(f"Could not find files for pretraining: `{pretraining}`, downstream: `{downstream}`, sample_size: `{sample_size}`")
        return None
    if len(files) != 5:
        print(f"Could not find all files for pretraining: `{pretraining}`, downstream: `{downstream}`, sample_size: `{sample_size}` ({len(files)}/5)")
    scores = []
    for file in files:
        scores.append(load_file(file))
    scores = [value[args.metric] for value in scores]
    return scores

def main():
    pretraining_datasets = ["STED", "SIM", "HPA", "JUMP", "ImageNet"]
    downstream_datasets = ["factin", "synaptic-semantic-segmentation", "footprocess", "lioness"]
    sample_size = ["50", "25", "10", "1"] if args.sampling_mode == "labels" else ["100", "50", "25", "10"]
    P, D = len(sample_size), len(downstream_datasets)
    performance_heatmap = np.zeros((P, D))
    text_heatmap = [[0] * D for _ in range(P)]
    for i, sample in tqdm(enumerate(sample_size), total=len(sample_size)):
        for j, downstream in enumerate(downstream_datasets):
            sample_performance = {}
            for k, pretraining in enumerate(pretraining_datasets):
                scores = get_data(pretraining=pretraining, downstream=downstream, sample_size=sample)
                scores_masked = np.ma.masked_equal(scores, -1)
                mean = np.ma.mean(scores_masked, axis=1)
                mean = np.mean(np.mean(mean, axis=-1))
                sample_performance[pretraining] = mean
                
            max_key = max(sample_performance.items(), key=lambda x: x[1])[0]
            performance_heatmap[i, j] = color_keys[max_key]
            text_heatmap[i][j] = max_key

    fig = plt.figure()
    ax = fig.add_subplot(111)
    print(np.unique(performance_heatmap))
    im = ax.imshow(performance_heatmap, cmap=custom_cmap, vmin=0, vmax=len(colors)-1)

    for i in range(P):
        for j in range(D):
            text = text_heatmap[i][j]
            ax.text(j, i, text, ha='center', va='center', color="black")
    ax.set_xticks(np.arange(D))
    ax.set_yticks(np.arange(P))
    ax.set_xticklabels(downstream_datasets)
    ylabels = [f"{int(i)}%" for i in sample_size]
    ax.set_yticklabels(ylabels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    savefig(fig, os.path.join(".", "results", f"{args.model}_{args.mode}_{args.sampling_mode}_small_heatmap"), extension="pdf")

if __name__=="__main__":
    main()



