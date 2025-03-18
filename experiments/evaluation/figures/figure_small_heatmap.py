import numpy as np 
import os 
import json 
import glob 
from tqdm import tqdm
import argparse 
import sys 
import matplotlib.pyplot as plt 
import matplotlib
sys.path.insert(0, "../../")
from DEFAULTS import BASE_PATH, COLORS 
from utils import savefig

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mae-small")
parser.add_argument("--mode", type=str, default="linear-probe")
parser.add_argument("--metric", type=str, default="acc")
args = parser.parse_args()


colors = ["white", "#5F4690", "#1D6996", "#0F8554", "#EDAD08", "#CC503E"]
color_keys = {
    "white": 0,
    "ImageNet": 1,
    "JUMP": 2,
    "HPA": 3,
    "SIM": 4,
    "STED": 5,
}


custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom", colors)

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
    

def main():
    sample_size = ["100", "50", "25", "10"]
    downstream_datasets = ["optim", "neural-activity-states", "peroxisome", "polymer-rings", "dl-sim"]
    P, D = len(sample_size), len(downstream_datasets)
    performance_heatmap = np.zeros((P, D))
    text_heatmap = [[0] * D for _ in range(P)]
    for i, sample in tqdm(enumerate(sample_size), total=len(sample_size)):
        for j, downstream in enumerate(downstream_datasets):
            sample_performance = {}
            if sample == "100" and downstream == "peroxisome":
                performance_heatmap[i, j] = color_keys["white"]
                text_heatmap[i][j] = "N/A"
            else:
                for k, pretraining in enumerate(["STED", "SIM", "HPA", "JUMP", "ImageNet"]):
                    scores = get_data(sample_size=sample, pretraining=pretraining, downstream=downstream, mode=args.mode)
                    sample_performance[pretraining] = np.mean(scores)
        
                max_key = max(sample_performance.items(), key=lambda x: x[1])[0]
                performance_heatmap[i, j] = color_keys[max_key]
                text_heatmap[i][j] = max_key
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(performance_heatmap, cmap=custom_cmap, vmin=0, vmax=len(colors)-1)

    for i in range(P):
        for j in range(D):
            text = text_heatmap[i][j]
            ax.text(j, i, text, ha='center', va='center', color="black")
    ax.set_xticks(np.arange(D))
    ax.set_yticks(np.arange(P))
    ax.set_xticklabels(downstream_datasets)
    ax.set_yticklabels(sample_size)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    savefig(fig, os.path.join(".", "results", f"{args.model}_{args.mode}_small_heatmap"), extension="pdf")
        


if __name__ == "__main__":
    main()
