import pickle
import os
import glob
import numpy

from matplotlib import pyplot
from collections import defaultdict

import sys
sys.path.insert(0, "../../")
from utils import savefig

MODELS = ["ImageNet", "JUMP", "HPA", "STED"]
CLASSES = [
    "PSD95", "Tubulin", "F-Actin", "Other"
]
CONVERT = {
    "f-actin": "F-Actin",
    "psd95": "PSD95",
    "tubulin": "Tubulin",
    "beta-camkii": "Other",
    "tom20" : "Other",
}
COLORS = {
    "ImageNet" : "tab:red",
    "JUMP" : "tab:green",
    "HPA" : "tab:orange",
    "STED" : "tab:blue"
}

def get_user(filename):
    basename = os.path.basename(filename)
    basename = os.path.splitext(basename)[0]
    return basename

def get_class(filename):
    basename = os.path.basename(filename)
    basename = os.path.splitext(basename)[0]
    class_id = basename.split("_")[-1]
    return CONVERT[class_id]
    # return CLASSES.index(CONVERT[class_id])

def get_model(filename):
    basename = os.path.basename(filename)
    basename = os.path.splitext(basename)[0]
    for model in MODELS:
        if model in basename.split("template")[0]:
            return model

def get_user_choices():
    files = glob.glob("data/*.pkl")
    per_user_scores = {}
    for file in files:
        scores = {c : [] for c in MODELS}
        with open(file, "rb") as f:
            data = pickle.load(f)
            user_choices = data["user_choices"]
        
        if len(user_choices) == 0:
            continue

        for key, value in user_choices.items():
            scores[get_model(key)].append({
                "truth" : get_class(key),
                "choice" : value,
                "model" : get_model(key)
            })
        per_user_scores[get_user(file)] = scores
    return per_user_scores


def main():
    user_choices = get_user_choices()
    
    accuracies = defaultdict(list)

    for user, choices in user_choices.items():
        for model, data in choices.items():

            confusion_matrix = numpy.zeros((len(CLASSES), len(CLASSES)))
            for d in data:
                confusion_matrix[CLASSES.index(d["truth"])][CLASSES.index(d["choice"])] += 1

            cm = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

            fig, ax = pyplot.subplots(figsize=(3,3))
            ax.imshow(cm, cmap="Purples")
            for j in range(len(CLASSES)):
                for i in range(len(CLASSES)):
                    ax.text(i, j, f"{cm[j, i]:.2f}", ha="center", va="center", color="black" if cm[j, i] < 0.5 else "white")
            ax.set_xticks(numpy.arange(len(CLASSES)))
            ax.set_yticks(numpy.arange(len(CLASSES)))
            ax.set_xticklabels(CLASSES)
            ax.set_yticklabels(CLASSES)
            pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            savefig(fig, f"./results/confusion_matrix-{user}-{model}", save_white=True)

            accuracies[model].append(numpy.diag(cm).mean())
    
    fig, ax = pyplot.subplots(figsize=(3,3))
    for model, values in accuracies.items():
        bplot = ax.boxplot(values, positions=[MODELS.index(model)], showfliers=True, widths=0.8)
        ax.scatter([MODELS.index(model)] * len(values), values, color="black")
        for name, parts in bplot.items():
            for part in parts:
                part.set_color(COLORS[model])
    ax.set(
        xticks=numpy.arange(len(MODELS)),
        xticklabels=MODELS,
        ylabel="Accuracy"
    )
    pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    savefig(fig, f"./results/accuracies", save_white=True)

if __name__ == "__main__":
    main()