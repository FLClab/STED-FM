
import os
import argparse
import itertools
import numpy
import torch
import glob
import networkx
import random

from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cosine
from matplotlib import pyplot

import sys 
sys.path.insert(0, "..")
from DEFAULTS import BASE_PATH, COLORS, DATASETS
from model_builder import get_pretrained_model_v2

FOLDER_NAMES = {
    "MAE_SMALL_IMAGENET1K_V1" : "mae-small_ImageNet",
    "MAE_SMALL_JUMP" : "mae-small_JUMP",
    "MAE_SMALL_HPA" : "mae-small_HPA",
    "MAE_SMALL_SIM" : "mae-small_SIM",
    "MAE_SMALL_STED" : "mae-small_STED"
}
def get_folder_name(weight):
    return FOLDER_NAMES[weight]

parser = argparse.ArgumentParser()   
parser.add_argument("--model", type=str, default="mae-lightning-small",
                    help="model model to load")
parser.add_argument("--seed", action="store_true", default=44,
                    help="Random seed")
parser.add_argument("--finetuned", action="store_true",
                    help="Use finetuned models")
parser.add_argument("--dataset", type=str, default="optim",
                    help="Dataset to use")
parser.add_argument("--metric", type=str, default="wasserstein", choices=["wasserstein", "cosine"],
                    help="Metric to use")
parser.add_argument("--figure", action="store_true",
                    help="Plot figures")
parser.add_argument("--opts", nargs="+", default=[], 
                    help="Additional configuration options")
parser.add_argument("--dry-run", action="store_true",
                    help="Activates dryrun")        
args = parser.parse_args()

def compute_distance(parameter_a, parameter_b, metric="wasserstein"):
    if metric == "wasserstein":
        return wasserstein_distance(parameter_a, parameter_b)
    elif metric == "cosine":
        if numpy.linalg.norm(parameter_a) == 0 or numpy.linalg.norm(parameter_b) == 0:
            return 0
        return cosine(parameter_a, parameter_b)
    else:
        raise ValueError(f"Unknown metric {metric}")

def load_models(weights):
    models = {}
    for weight in weights:
        n_channels = 1
        if weight is not None:
            n_channels = 3 if "imagenet" in weight.lower() else n_channels

        model, cfg = get_pretrained_model_v2(
            name=args.model,
            weights=weight,
            path=None,
            mask_ratio=0.0, 
            pretrained=True if n_channels==3 else False,
            in_channels=n_channels,
            as_classifier=True,
            blocks="all",
            num_classes=1
        )

        models[weight] = model
    return models

def load_models_finetuned(weights):
    models = {}
    for weight in weights:
        n_channels = 1
        if weight is not None:
            n_channels = 3 if "imagenet" in weight.lower() else n_channels

        checkpoint = torch.load(
            os.path.join(
                BASE_PATH, "baselines", get_folder_name(weight), args.dataset, f"finetuned_None_{args.seed}.pth"
            )
        )
        model, cfg = get_pretrained_model_v2(
            name=args.model,
            weights=weight,
            path=None,
            mask_ratio=0.0, 
            pretrained=True if n_channels==3 else False,
            in_channels=n_channels,
            as_classifier=True,
            blocks="all",
            num_classes=checkpoint["model_state_dict"]["classification_head.1.weight"].shape[0]
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        models[weight] = model
    return models

def plot_graphs():
    files = glob.glob("features/model-weights/weights.npz")

    for file in files:
        G = networkx.Graph()
        data = numpy.load(file)

        for key, values in data.items():
            model_a, model_b = key.split(";")
            if args.finetuned and ("finetuned" not in model_a or "finetuned" not in model_b):
                continue
            elif not args.finetuned and ("finetuned" in model_a or "finetuned" in model_b):
                continue
                
            values = values[values < 1]
            auc = numpy.trapz(numpy.arange(len(values)) / len(values), numpy.sort(values))
            G.add_edge(model_a, model_b, weight=1 - auc)
    
        fig, ax = pyplot.subplots()
        networkx.draw_kamada_kawai(
            G, with_labels=True, ax=ax,
            node_color=[COLORS[node] for node in G.nodes],
            labels={node: DATASETS[node] for node in G.nodes},
        )
        basename = os.path.basename(file)
        basename = basename.replace(".npz", ".pdf")
        savedir = "figures/model-weights"
        os.makedirs(savedir, exist_ok=True)
        fig.savefig(f"{savedir}/{basename}")

def plot_cumulative_layer():
    files = glob.glob("features/model-weights/weights.npz")

    for file in files:
        data = numpy.load(file)
        fig, ax = pyplot.subplots()
        for key, values in data.items():
            model_a, model_b = key.split(";")
            if args.finetuned and ("finetuned" not in model_a or "finetuned" not in model_b):
                continue
            elif not args.finetuned and ("finetuned" in model_a or "finetuned" in model_b):
                continue
                
            ax.plot(numpy.cumsum(values) / numpy.sum(values), label=f"{DATASETS[model_a]} vs {DATASETS[model_b]}")
            # ax.plot(numpy.cumsum(values), label=f"{DATASETS[model_a]} vs {DATASETS[model_b]}")
        ax.set(
            xlabel="Layers",
            ylabel="Cumulative distance"
        )
        ax.legend()

        basename = os.path.basename(file)
        basename = basename.replace(".npz", "_per-layer.pdf")
        savedir = "figures/model-weights"
        os.makedirs(savedir, exist_ok=True)
        fig.savefig(f"{savedir}/{basename}")

def main():

    random.seed(args.seed)
    numpy.random.seed(args.seed)

    if args.figure:
        plot_graphs()
        plot_cumulative_layer()
        return

    weights = [
        ("MAE_SMALL_IMAGENET1K_V1", False),
        ("MAE_SMALL_IMAGENET1K_V1", True),
        ("MAE_SMALL_JUMP", False),
        ("MAE_SMALL_JUMP", True),
        ("MAE_SMALL_HPA", False),
        ("MAE_SMALL_HPA", True),
        ("MAE_SMALL_SIM", False),
        ("MAE_SMALL_SIM", True), 
        ("MAE_SMALL_STED", False),
        ("MAE_SMALL_STED", True),
    ]

    _weights = [w[0] for w in weights if w[1] == args.finetuned]
    finetuned_models = load_models_finetuned(_weights)
    models = load_models(_weights)


    output = {}

    fig, ax = pyplot.subplots()
    bins = numpy.linspace(0, 1, 250)
    for weight_a, weight_b in itertools.combinations(weights, 2):


        weight_a, finetuned_a = weight_a
        weight_b, finetuned_b = weight_b

        print(f"Comparing {weight_a} vs {weight_b}")
        if finetuned_a:
            model_a = finetuned_models[weight_a]
        else:
            model_a = models[weight_a]
        
        if finetuned_b:
            model_b = finetuned_models[weight_b]
        else:
            model_b = models[weight_b]

        distances = []
        keys = model_a.state_dict().keys()
        for key, parameter_a in model_a.named_parameters():

            if "classification_head" in key:
                continue

            parameter_b = model_b.state_dict()[key]
            if parameter_a.shape != parameter_b.shape:
                print(f"Shapes do not match for {key}")
                print(parameter_a.shape, parameter_b.shape)
                continue

            parameter_a = parameter_a.detach().cpu().numpy().ravel()
            parameter_b = parameter_b.detach().cpu().numpy().ravel()


            distance = compute_distance(parameter_a, parameter_b, metric=args.metric)
            distances.append(distance)
        
        argsort = numpy.argsort(distances)
        for idx in argsort[-10:]:
            print(f"Layer: {list(keys)[idx]} Distance: {distances[idx]}")

        # ax.plot(distances, label=f"{weight_a} vs {weight_b}")
        ax.hist(distances, cumulative=True, density=True, bins=bins, alpha=0.5, histtype="step", label=f"{weight_a} vs {weight_b}")
        fig.savefig("weights.pdf")

        weight_a = weight_a + "_finetuned" if finetuned_a else weight_a
        weight_b = weight_b + "_finetuned" if finetuned_b else weight_b
        output[";".join([weight_a, weight_b])] = distances

    ax.set(
        xlabel=args.metric,
        ylabel="Frequency"
    )
    ax.legend()
    fig.savefig("weights.pdf")

    os.makedirs("features/model-weights", exist_ok=True)
    savename = "features/model-weights/weights.npz"
    if args.finetuned:
        savename = "features/model-weights/weights-finetuned.npz"
    numpy.savez(savename, **output)

if __name__ == "__main__":

    main()