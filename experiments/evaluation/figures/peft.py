import numpy as np 
import matplotlib.pyplot as plt 
import json 
import argparse 
import sys 
import glob
import os
from scipy import stats 
from stedfm.DEFAULTS import BASE_PATH, COLORS, MARKERS 
from stedfm.utils import savefig

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mae-small")
parser.add_argument("--dataset", type=str, default="optim")
parser.add_argument("--metric", type=str, default="acc")
args = parser.parse_args() 

def load_file(file):
    with open(file, "r") as handle:
        data = json.load(handle)
    return data 

def get_data(pretraining: str, downstream: str, mode: str) -> dict:
    files = glob.glob(os.path.join(BASE_PATH, "baselines", f"{args.model}_{pretraining}", downstream, f"accuracy_{mode}_None_*.json"), recursive=True)
    if len(files) < 1: 
        print(f"Could not find files for mode: `{mode}` and pretraining: `{pretraining}`")
        exit()
    if len(files) > 5:
        print(f"Found more than 5 files for mode: `{mode}` and pretraining: `{pretraining}`")
        exit()
    scores = [load_file(file) for file in files]
    scores = [value[args.metric] for value in scores]
    return scores

def get_peft_data(pretraining: str, downstream: str, num_blocks: int) -> dict:
    files = glob.glob(os.path.join(BASE_PATH, "baselines", f"{args.model}_{pretraining}", downstream, f"accuracy_peft_{num_blocks}-blocks_None_*.json"), recursive=True)
    if len(files) < 1: 
        print(f"Could not find files for mode: `peft_{num_blocks}-blocks` and pretraining: `{pretraining}`")
        exit()
    if len(files) > 5:
        print(f"Found more than 5 files for mode: `peft_{num_blocks}-blocks` and pretraining: `{pretraining}`")
        exit()
    scores = [load_file(file) for file in files]
    scores = [value[args.metric] for value in scores]
    return scores

def main():
    pretraining_datasets = ["STED", "SIM", "HPA", "JUMP", "ImageNet"]
    num_blocks = ["10", "8", "6", "4", "2"]


    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = [0, 2, 4, 6, 8, 10, 12]
    for pretraining in pretraining_datasets:
        peft_curve = []

        linear_probing_data = np.mean(get_data(pretraining=pretraining, downstream=args.dataset, mode="linear-probe"))
        peft_curve.append(linear_probing_data)
        

        for nb in num_blocks:
            peft_data = np.mean(get_peft_data(pretraining=pretraining, downstream=args.dataset, num_blocks=nb))
            peft_curve.append(peft_data)

        finetuning_data = np.mean(get_data(pretraining=pretraining, downstream=args.dataset, mode="finetuned")) 
        peft_curve.append(finetuning_data) 

        ax.plot(x, peft_curve, label=pretraining, marker=MARKERS[pretraining], color=COLORS[pretraining])

    ax.set_xlabel("# Fine-tuned blocks")
    ax.set_ylabel(args.metric)
    ax.legend()
    savefig(fig, os.path.join(".", "results", f"peft_{args.model}_{args.dataset}"), extension="pdf")
        





if __name__ == "__main__":
    main()