import matplotlib.pyplot as plt 
import numpy as np
import glob
import argparse 
from typing import Dict
import sys 
sys.path.insert(0, "../../")
from DEFAULTS import COLORS

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mae-lightning-small")
args = parser.parse_args()

def load_data(path: str, pretraining: str) -> np.ndarray:
    files = glob.glob(f"{path}/*.npz")
    files = [f for f in files if pretraining in f]
    return list(set(files))

def plot_results(results: Dict) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    datasets = list(results["IMAGENET1K_V1"].keys())
    width = 0.2
    group_gap = 0.2  # Gap between groups of bars
    
    # Calculate the total width of one group of bars
    n_bars = len(results)
    total_width = n_bars * width
    
    # Calculate starting positions for each group
    group_positions = np.arange(len(datasets)) * (total_width + group_gap)
    
    for idx, (key, values) in enumerate(results.items()):
        data = [accuracies[0] for accuracies in values.values()]
        # Position each bar within its group
        x_positions = group_positions + (idx * width)
        bars = ax.bar(x_positions, data, width=width, label=key)
        for bar in bars:
            bar.set_color(COLORS[key])
    
    # Center the xticks in each group
    ax.set_xticks(group_positions + total_width/2 - width/2)
    ax.set_xticklabels(datasets, rotation=45)
    fig.savefig(f"./temp.png")


if __name__=="__main__":
    pretrainings = ["IMAGENET1K_V1", "JUMP", "HPA", "SIM", "STED"]
    results = {key: None for key in pretrainings}
    for pretraining in pretrainings:
        pretraining_results = {key: None for key in ["optim", "neural-activity-states", "peroxisome", "polymer-rings", "dl-sim"]}
        files = load_data(path=f"../results/{args.model}", pretraining=pretraining)
        for f in files:
            dataset = f.split("/")[-1].split("_")[0]
            data = np.load(f)["accuracies"].tolist()
            print(f" === {dataset} | {pretraining} === ")
            print(f"\t{data}")
            print("\n")
            pretraining_results[dataset] = data
        results[pretraining] = pretraining_results

    


        
        