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
from stedfm.DEFAULTS import BASE_PATH, COLORS
from stedfm.utils import savefig 

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--metric", type=str, default="iou")
parser.add_argument("--mode", type=str, default="pretrained-frozen")
parser.add_argument("--dataset", type=str, default="footprocess")
args = parser.parse_args()

def load_file(file):
    with open(file, "r") as handle:
        data = json.load(handle)
    return data 

def get_data(pretraining: str, downstream: str, mode: str):
    if mode != "from-scratch":
        files = glob.glob(os.path.join(BASE_PATH, "segmentation-baselines", f"{args.model}", downstream, f"{mode}*_{pretraining.upper()}*", f"segmentation-scores.json"), recursive=True)
        if args.mode == "pretrained-frozen":
            files = [f for f in files if "frozen" in f]
        if args.mode == "pretrained":
            files = [f for f in files if "frozen" not in f]
    else:
        files = glob.glob(os.path.join(BASE_PATH, "segmentation-baselines", f"{args.model}", downstream, f"{mode}*", f"segmentation-scores.json"), recursive=True)

    files = [f for f in files if "labels" not in f]
    files = [f for f in files if "samples" not in f]

    if len(files) < 1:
        print(f"Could not find files for pretraining: `{pretraining}`, downstream: `{downstream}`")
        return data
    if len(files) != 5:
        print(f"Could not find all files for pretraining: `{pretraining}`, downstream: `{downstream}` ({len(files)}/5)")
    scores = [] 
    for file in files:
        scores.append(load_file(file))
    scores = [value[args.metric] for value in scores]
    return scores

def main():
    pretraining_dataset = "STED"

    scores = get_data(pretraining=pretraining_dataset, downstream=args.dataset, mode=args.mode)
    class_scores = np.zeros((len(scores), 2))
    for i, s in enumerate(scores):
        s_masked = np.ma.masked_equal(s, -1)
        mean = np.ma.mean(s_masked, axis=0)
        class_scores[i] = mean 

    for c in range(class_scores.shape[1]):
        print(f"Class {c} average: {class_scores[:, c].mean()}")

if __name__=="__main__":
    main()