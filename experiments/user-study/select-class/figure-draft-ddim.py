import pickle
import os
import glob
import numpy

from matplotlib import pyplot
from collections import defaultdict

from stedfm.DEFAULTS import COLORS
from stedfm.utils import savefig

class User:
    def __init__(self, name):
        self.name = name

EXPERIMENT = "draft-ddim-v2"

MODELS = ["ImageNet", "JUMP", "HPA", "SIM", "STED", "classifier", "real"]
MODELS = ["ddim", "draft"]

# Manually adding the colors for the models
COLORS["classifier"] = "silver"
COLORS["real"] = "silver"
COLORS["ddim"] = "tab:blue"
COLORS["draft"] = COLORS["STED"]

if EXPERIMENT == "draft-ddim":
    CLASSES = [
        "Odd", "Blurry", "Accurate", "Artefact", "No rings but relevant"
    ]
elif EXPERIMENT == "draft-ddim-v2":
    CLASSES = [
        "Rings", "Other structures", "Nothing"
    ]

CONVERT = {
    "f-actin": "F-Actin",
    "PSD95": "PSD95",
    "tubulin": "Tubulin",
    "beta-camkii": "Other",
    "vglut2" : "Other",
    "tom20" : "Other",
    "ddim" : "DDIM",
    "draft" : "DRaFT"
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
    if len(basename.split("template")[0]) == 0:
        return "real"
    for model in MODELS:
        if model.lower() in basename.split("_")[-1].lower():
            return model
    
    print(f"Model not found for {basename}")

def get_user_choices():
    files = glob.glob(f"data/{EXPERIMENT}/*.pkl")
    per_user_scores = {}
    for file in files:
        scores = {c : [] for c in MODELS}
        with open(file, "rb") as f:
            data = pickle.load(f)
            user_choices = data["user_choices"]
        print(user_choices)
        if len(user_choices) == 0:
            continue

        for key, value in user_choices.items():
            model = get_model(key)
            if model is None:
                continue
            scores[get_model(key)].append({
                "truth" : get_class(key),
                "choice" : value,
                "model" : get_model(key)
            })
        per_user_scores[get_user(file)] = scores
    return per_user_scores

def count_per_class(data):
    counts = numpy.zeros(len(CLASSES))
    for d in data:
        counts[CLASSES.index(d["choice"])] += 1
    return counts

def main():
    user_choices = get_user_choices()
    print(user_choices)
    
    choices_per_model = defaultdict(list)
    for user, choices in user_choices.items():
        for model, data in choices.items():
            counts = count_per_class(data)
            choices_per_model[model].append(counts)
    print(choices_per_model)
    fig, ax = pyplot.subplots(figsize=(3,3))
    for idx, (model, counts) in enumerate(choices_per_model.items()):
        counts = numpy.array(counts)
        mean_counts = counts.mean(axis=0)
        positions = numpy.arange(len(CLASSES)) + idx * 0.4
        for c in counts:
            ax.scatter(positions, c, color="black", alpha=0.5, zorder=100)
        ax.bar(positions, mean_counts, color=COLORS[model], label=model, width=0.4, edgecolor="black")

    ax.set(
        xticks=numpy.arange(len(CLASSES)) + 0.4 / 2,
        xticklabels=CLASSES,
        ylabel="Count"
    )
    pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    os.makedirs(f"./results/{EXPERIMENT}", exist_ok=True)
    savefig(fig, f"./results/{EXPERIMENT}/accuracies", save_white=True)
    exit()
    # fig, ax = pyplot.subplots(figsize=(3,3))
    # to_plot = []
    # for model, values in unclassiables.items():
    #     to_plot.append(values)
    # to_plot = numpy.array(to_plot)
    # print(to_plot.shape)
    # # for userdata in to_plot:
    # #     ax.plot(userdata[0], userdata[1], color="silver")
    # # ax.set(
    # #     xticks=numpy.arange(len(MODELS)),
    # #     xticklabels=MODELS,
    # #     ylabel="Unclassifiable"
    # # )
    # pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # savefig(fig, f"./results/unclassifiables", save_white=True)

    samples = [values for values in accuracies.values()]

    from scipy.stats import kruskal
    import scikit_posthocs
    H, p_value = kruskal(*samples)
    if p_value < 0.05:
        print(f"Kruskal-Wallis test: Reject null hypothesis (p_value: {p_value})")
        result = scikit_posthocs.posthoc_mannwhitney(samples)
        print(result)

    from stedfm.stats import resampling_stats, plot_p_values
    p_values, F_p_value = resampling_stats(samples, labels=list(accuracies.keys()))
    print(p_values)
    print(F_p_value)
    fig, ax = plot_p_values(p_values)
    savefig(fig, f"./results/p_values", save_white=True)

if __name__ == "__main__":
    main()