import pickle
import os
import glob
import numpy

from matplotlib import pyplot
from collections import defaultdict

from stedfm.DEFAULTS import COLORS
from stedfm.utils import savefig

MODELS = ["ImageNet", "JUMP", "HPA", "SIM", "STED", "classifier", "real"]
MODELS = ["classifier", "STED", "real"]

# Manually adding the colors for the models
COLORS["classifier"] = "silver"
COLORS["real"] = "silver"

CLASSES = [
    "PSD95", "Tubulin", "F-Actin", "Other", "Unclassifiable"
]
CONVERT = {
    "f-actin": "F-Actin",
    "PSD95": "PSD95",
    "tubulin": "Tubulin",
    "beta-camkii": "Other",
    "vglut2" : "Other",
    "tom20" : "Other",
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
        if model.lower() in basename.split("template")[0].lower():
            return model
    
    print(f"Model not found for {basename}")

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


def main():
    user_choices = get_user_choices()
    
    accuracies = defaultdict(list)
    unclassiables = defaultdict(list)

    for user, choices in user_choices.items():
        for model, data in choices.items():

            confusion_matrix = numpy.zeros((len(CLASSES), len(CLASSES)))
            for d in data:
                try:
                    confusion_matrix[CLASSES.index(d["truth"])][CLASSES.index(d["choice"])] += 1
                except ValueError:
                    pass

            print(confusion_matrix)
            
            cm = confusion_matrix[:-2] / confusion_matrix[:-2].sum(axis=1, keepdims=True)
            confusion_matrix[:-2] = cm

            fig, ax = pyplot.subplots(figsize=(3,3))
            ax.imshow(confusion_matrix, cmap="Purples", vmin=0, vmax=1)
            for j in range(len(CLASSES)):
                for i in range(len(CLASSES)):
                    ax.text(i, j, f"{confusion_matrix[j, i]:.2f}", ha="center", va="center", color="black" if confusion_matrix[j, i] < 0.5 else "white")
            ax.set_xticks(numpy.arange(len(CLASSES)))
            ax.set_yticks(numpy.arange(len(CLASSES)))
            ax.set_xticklabels(CLASSES)
            ax.set_yticklabels(CLASSES)
            pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            savefig(fig, f"./results/confusion_matrix-{user}-{model}", save_white=True)
            pyplot.close("all")

            # Considers Other and Unclassifiable as the same class
            cm[:, -2] += cm[:, -1]
            print(cm)
            accuracies[model].append(numpy.diag(cm).mean())
            unclassiables[model].append(confusion_matrix[:-1, -1])

    fig, ax = pyplot.subplots(figsize=(3,3))
    to_plot = []
    for model, values in accuracies.items():
        bplot = ax.boxplot(values, positions=[MODELS.index(model)], showfliers=True, widths=0.8)
        ax.scatter([MODELS.index(model)] * len(values), values, color="black")
        to_plot.append([[MODELS.index(model)] * len(values), values])
        for name, parts in bplot.items():
            for part in parts:
                part.set_color(COLORS[model])
    to_plot = numpy.array(to_plot).swapaxes(2, 0)
    for userdata in to_plot:
        ax.plot(userdata[0], userdata[1], color="silver")
    ax.set(
        xticks=numpy.arange(len(MODELS)),
        xticklabels=MODELS,
        ylabel="Accuracy", 
        ylim=(0, 1)
    )
    pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    savefig(fig, f"./results/accuracies", save_white=True)

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