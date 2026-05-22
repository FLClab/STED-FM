
import pickle
import os
import glob
import numpy 

from matplotlib import pyplot

from stedfm.DEFAULTS import COLORS
from stedfm.utils import savefig

CLASSES = [
    "pix2pix",
    "ddim",
    "draft",
]
NAMES = {
    "ddim" : "DDIM",
    "draft" : "DRAFT",
    "pix2pix" : "Pix2Pix"
}
COLORS = {
    "ddim" : "tab:blue",
    "draft" : COLORS["STED"],
    "pix2pix" : "tab:green"
}

class User:
    def __init__(self, name):
        self.name = name

def simple_beeswarm(y, nbins=None, maxwidth=0.8):
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.
    """
    y = numpy.asarray(y)
    if nbins is None:
        nbins = len(y) // 6
        nbins = max(2, nbins)

    # Get upper bounds of bins
    x = numpy.zeros(len(y))
    ylo = numpy.min(y)
    yhi = numpy.max(y)
    dy = (yhi - ylo) / nbins
    ybins = numpy.linspace(ylo + dy, yhi - dy, nbins - 1)

    # Divide indices into bins
    i = numpy.arange(len(y))
    ibs = [0] * nbins
    ybs = [0] * nbins
    nmax = 0
    for j, ybin in enumerate(ybins):
        f = y <= ybin
        ibs[j], ybs[j] = i[f], y[f]
        nmax = max(nmax, len(ibs[j]))
        f = ~f
        i, y = i[f], y[f]
    ibs[-1], ybs[-1] = i, y
    nmax = max(nmax, len(ibs[-1]))

    # Assign x indices
    dx = 1 / (nmax // 2)
    for i, y in zip(ibs, ybs):
        if len(i) > 1:
            j = len(i) % 2
            i = i[numpy.argsort(y)]
            a = i[j::2]
            b = i[j+1::2]
            x[a] = (0.5 + j / 3 + numpy.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + numpy.arange(len(b))) * -dx
    
    x = x / numpy.max(numpy.abs(x)) * maxwidth / 2
    return x

def get_class(filename):
    basename = os.path.basename(filename)
    for c in CLASSES:
        if c in basename:
            return c
    return None

def get_user_choices():
    files = glob.glob("data/FActinDendriticDataset/*.pkl")
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
    files = glob.glob("data/FActinDendriticDataset/*.pkl")
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
    savefig(fig, "./results/FActinDendriticDataset/disagreement", save_white=True)
        
def main():

    numpy.random.seed(42)

    # get_selections()
    per_user_scores = get_user_choices()

    all_values = []
    for user, scores in per_user_scores.items():
        values = numpy.array([scores[c] for c in CLASSES])
        all_values.append(values)
    all_values = numpy.array(all_values)

    all_values = all_values / all_values.sum(axis=1, keepdims=True)

    fig, ax = pyplot.subplots(figsize=(3, 3))
    samples = []
    highlights = ["Antoine O"]
    highlights = []
    for i in range(all_values.shape[1]):
        mean = numpy.mean(all_values[:, i])
        std = numpy.std(all_values[:, i])
        xs = simple_beeswarm(all_values[:, i], maxwidth=0.3)
        print(per_user_scores.keys())
        edgecolor = ["black" if os.path.basename(user).split(".")[0] not in highlights else "silver" for user in per_user_scores.keys()]
        ax.scatter(i + xs, all_values[:, i], facecolor="none", edgecolor=edgecolor, zorder=100)
        ax.bar(i, mean, yerr=std, width=0.8, label=CLASSES[i], align="center", color=COLORS[CLASSES[i]])

        samples.append(all_values[:, i])
    
    ax.set_xticks(numpy.arange(len(scores.keys())))
    ax.set_xticklabels([NAMES[c] for c in CLASSES], rotation=45)
    ax.set(
        ylabel="Proportion (-)", ylim=(0, 1)
    )
    savefig(fig, "./results/FActinDendriticDataset/choices", save_white=True)

    print(samples)
    for sample, c in zip(samples, CLASSES):
        print(f"{c}: mean={numpy.mean(sample):.4f}, std={numpy.std(sample):.4f}")

    from scipy.stats import kruskal, mannwhitneyu

    print(f"Samples: {[len(s) for s in samples]}")
    print("Kruskal-Wallis H-test")
    hstat, pvalue = kruskal(*samples)
    print(f"H-statistic: {hstat:.4f}, p-value: {pvalue:.4e}")

    print("Mann-Whitney U test")
    for i in range(len(samples)-1):
        for j in range(i + 1, len(samples)):
            ustat, pvalue = mannwhitneyu(samples[i], samples[j])
            print(f"{CLASSES[i]} vs {CLASSES[j]}: U-statistic: {ustat:.4e}, p-value: {pvalue:.4e}")

    # from stedfm.stats import resampling_stats
    # print("Resampling test")
    # p_values, F_p_value = resampling_stats(samples, CLASSES)
    # print("Overall F-test p-value:")
    # print(F_p_value)
    # print("Pairwise p-values:")
    # print(p_values)

if __name__ == "__main__":
    main()