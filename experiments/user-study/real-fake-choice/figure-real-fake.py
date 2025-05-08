
import pickle
import os
import glob
import numpy 
import argparse

from matplotlib import pyplot

# import sys
# sys.path.insert(0, "../../")
# from DEFAULTS import COLORS
# from utils import savefig

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="example", help="Dataset to use")
args = parser.parse_args()

CLASSES = [
    "model-a",
    "model-b",
]

class User:
    def __init__(self, name):
        self.name = name

def get_color(model):
    for key, value in COLORS.items():
        if key.lower() in model.lower():
            return value
    return COLORS[model]

def get_class(filename):
    basename = os.path.basename(filename)
    for c in CLASSES:
        if c in basename:
            return c
    return None

def get_user_choices():
    files = glob.glob(f"data/{args.dataset}/*.pkl")
    per_user_scores = {}
    for file in files:
        scores = {c : 0 for c in CLASSES}
        num_candidates_per_class = {c : 0 for c in CLASSES}
        with open(file, "rb") as f:
            data = pickle.load(f)
            user_choices = data["user_choices"]
        if len(user_choices) == 0:
            continue
        for template, candidate, selected in user_choices:
            print(template, candidate, selected)
            scores[get_class(candidate)] += candidate == selected
            num_candidates_per_class[get_class(candidate)] += 1
        
        # Normalize scores
        for c in CLASSES:
            if num_candidates_per_class[c] > 0:
                scores[c] = scores[c] / num_candidates_per_class[c]
            else:
                scores[c] = 0
        per_user_scores[file] = scores

    return per_user_scores

def merge_dicts(dicts):
    merged = {}
    for i, d in enumerate(dicts):
        # print(i)
        for key, value in d.items():
            # print(os.path.basename(key) in value)
            if key not in merged:
                merged[key] = [value]
            else:
                merged[key].append(value)
    return merged

def get_selections():
    files = glob.glob(f"data/{args.dataset}/*.pkl")
    per_user_data = []
    for file in files:
        with open(file, "rb") as f:
            data = pickle.load(f)
            per_user_data.append(data["user_choices"])
    return per_user_data
        
def main():

    numpy.random.seed(42)

    per_user_scores = get_user_choices()
    print(per_user_scores)

if __name__ == "__main__":
    main()