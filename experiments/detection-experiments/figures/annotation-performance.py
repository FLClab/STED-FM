
import os
import glob
import numpy
import pickle
import matplotlib.ticker as mticker

from matplotlib import pyplot

from stedfm.DEFAULTS import COLORS
from stedfm.utils import savefig

PATH = "../outputs/"

def load_history_of_models(args):
    history_of_models = {}
    for model in args.models:
        model_path = os.path.join(PATH, "best_rf_model-history-{}.pkl".format(model))
        with open(model_path, "rb") as f:
            history_of_models[model] = pickle.load(f)
    return history_of_models

def load_model_scores(args):
    scores = {}
    for model in args.models:
        model_path = os.path.join(PATH, "scores_per_model_rf-{}.pkl".format(model))
        with open(model_path, "rb") as f:
            scores[model] = pickle.load(f)
    return scores

def plot_annotation_performance(keep_indices, model_scores, model_history, metric="f1", figax=None):

    if figax is not None:
        fig, ax = figax
    else:
        fig, ax = pyplot.subplots(figsize=(3,3))

    xs = numpy.array([history["X_train"].shape[0] for history in model_history])

    xs = xs * (16 * 16) / (625 * 625)

    ys = numpy.array([score[metric] for score in model_scores])

    mean, std = numpy.mean(ys[:, keep_indices], axis=1), numpy.std(ys[:, keep_indices], axis=1)

    ax.plot(xs, mean, color="silver", alpha=0.3, zorder=1)
    # ax.fill_between(xs, mean - std, mean + std, color="silver", alpha=0.5)

    return fig, ax, (xs, mean)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="+", required=True)
    args = parser.parse_args()

    history_of_models = load_history_of_models(args)
    scores = load_model_scores(args)
    
    for metric in ["f1", "precision", "recall"]:
        fig, ax = pyplot.subplots(figsize=(3,3))
        interpolated_xs = numpy.linspace(0.01, 0.20, num=50)
        means = []
        for model_idx, model_scores in scores.items():
            model_history = history_of_models[model_idx]
            keep_indices = list(set(range(len(args.models))) - {model_idx})
            fig, ax, (xs, mean) = plot_annotation_performance(keep_indices, model_scores, model_history, metric=metric, figax=(fig, ax))

            interpolated_mean = numpy.interp(interpolated_xs, xs, mean)
            means.append(interpolated_mean)

        means = numpy.array(means)
        overall_mean, overall_std = numpy.mean(means, axis=0), numpy.std(means, axis=0)
        ax.plot(interpolated_xs, overall_mean, color="tab:blue", alpha=1.0)
        ax.fill_between(interpolated_xs, overall_mean - overall_std, overall_mean + overall_std, color="tab:blue", alpha=0.3)
        ax.set(
            xlabel="Annotated Fraction (%)",
            ylabel=f"{metric.capitalize()} Score",
            # xscale="log",
            ylim=(0.5, 1)
        )

        formatter = mticker.PercentFormatter(xmax=1.0, decimals=0) # decimals=0 for no decimal places
        ax.xaxis.set_major_formatter(formatter)
        savefig(fig, f"./figures/annotation_performance_overall_{metric}", dpi=300)
        pyplot.close(fig)

if __name__ == "__main__":
    main()