
import pickle
import os
import glob
import numpy 

from matplotlib import pyplot

import sys
sys.path.insert(0, "../../")
from DEFAULTS import COLORS
from utils import savefig

CLASSES = [
    "MAE_SMALL_IMAGENET1K_V1",
    "MAE_SMALL_JUMP",
    "MAE_SMALL_HPA",
    "MAE_SMALL_SIM",
    "MAE_SMALL_STED"
]
NAMES = {
    "MAE_SMALL_IMAGENET1K_V1" : "ImageNet",
    "MAE_SMALL_JUMP" : "JUMP",
    "MAE_SMALL_HPA" : "HPA",
    "MAE_SMALL_SIM" : "SIM",
    "MAE_SMALL_STED" : "STED"
}

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
    files = glob.glob("data/attention-maps/*.pkl")
    per_user_scores = {}
    for file in files:
        scores = {c : 0 for c in CLASSES}
        with open(file, "rb") as f:
            data = pickle.load(f)
            user_choices = data["user_choices"]
        if len(user_choices) == 0:
            continue
        for key, value in user_choices.items():
            scores[get_class(value)] += 1
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
    files = glob.glob("data/attention-maps/*.pkl")
    per_user_data = []
    for file in files:
        with open(file, "rb") as f:
            data = pickle.load(f)

            # Makes sure that the user choices are consistent
            to_remove = []
            for key, value in data["user_choices"].items():
                if not (os.path.basename(key) in value):
                    to_remove.append(key)
            if to_remove:
                print(f"Removing {len(to_remove)} inconsistent selections for user {data['user'].name}")
            for key in to_remove:
                del data["user_choices"][key]

            per_user_data.append(data["user_choices"])
    
    merged = merge_dicts(per_user_data)

    largest_set = max([len(set(values)) for values in merged.values()])
    for key, values in merged.items():
        print(len(set(values)), set(values))

        # Every user selected the same image
        if len(set(values)) == 1:
            print(os.path.basename(key), NAMES[get_class(values[0])])
        elif len(set(values)) == largest_set:
            print(os.path.basename(key), [NAMES[get_class(v)] for v in values])

    # Rank by disagreement
    disagreements = {key : len(set(values)) for key, values in merged.items()}
    values = []
    for key, value in sorted(disagreements.items(), key=lambda x: x[1]):
        values.append(value)

    fig, ax = pyplot.subplots(figsize=(3, 3))
    ax.plot(values)
    ax.set(
        ylabel="Disagreement (-)"
    )
    savefig(fig, "./results/attention-maps/disagreement", save_white=True)
        
def main():

    numpy.random.seed(42)

    get_selections()

    per_user_scores = get_user_choices()

    all_values = []
    for user, scores in per_user_scores.items():
        values = numpy.array([scores[c] for c in CLASSES])
        all_values.append(values)
    all_values = numpy.array(all_values)

    all_values = all_values / all_values.sum(axis=1, keepdims=True)

    fig, ax = pyplot.subplots(figsize=(3, 3))
    for i in range(all_values.shape[1]):
        mean = numpy.mean(all_values[:, i])
        std = numpy.std(all_values[:, i])
        ax.scatter(numpy.random.normal(i, 0., size=all_values.shape[0]), all_values[:, i], facecolor="none", edgecolor="black", zorder=100)
        ax.bar(i, mean, yerr=std, width=0.8, label=CLASSES[i], align="center", color=COLORS[CLASSES[i]])
    
    ax.set_xticks(numpy.arange(len(scores.keys())))
    ax.set_xticklabels([NAMES[c] for c in CLASSES], rotation=45)
    ax.set(
        ylabel="Proportion (-)", ylim=(0, 1)
    )
    savefig(fig, "./results/attention-maps/choices", save_white=True)

if __name__ == "__main__":
    main()