
import numpy
from matplotlib import pyplot

from scipy.stats import pearsonr
from stedfm.DEFAULTS import COLORS

scores = {
    "STED" : {
        "SO" : 0.96,
        "NAS" : 0.45,
        "Px" : 0.67,
        "PR" : 0.88,
        "DL-SIM" : 0.95,
        "BBBC026" : 0.88,
        "BBBC052" : 0.86,
        "BBBC053" : 0.81,
        "F-Actin" : 0.65,
        "SPZ" : 0.83, 
        "FP" : 0.45,
        "Lioness" : 0.71
    },
    "SIM" : {
        "SO" : 0.91,
        "NAS" : 0.33,
        "Px" : 0.64,
        "PR" : 0.86,
        "DL-SIM" : 0.97,
        "BBBC026" : 0.84,
        "BBBC052" : 0.78,
        "BBBC053" : 0.79,        
        "F-Actin" : 0.58,
        "SPZ" : 0.80, 
        "FP" : 0.40,
        "Lioness" : 0.71        
    },
    "HPA" : {
        "SO" : 0.93,
        "NAS" : 0.25,
        "Px" : 0.65,
        "PR" : 0.84,
        "DL-SIM" : 0.91,
        "BBBC026" : 0.87,
        "BBBC052" : 0.78,
        "BBBC053" : 0.77,        
        "F-Actin" : 0.61,
        "SPZ" : 0.80, 
        "FP" : 0.42,
        "Lioness" : 0.71
    },
    "JUMP" : {
        "SO" : 0.81,
        "NAS" : 0.34,
        "Px" : 0.61,
        "PR" : 0.85,
        "DL-SIM" : 0.94,
        "BBBC026" : 0.91,
        "BBBC052" : 0.76,
        "BBBC053" : 0.80,        
        "F-Actin" : 0.59,
        "SPZ" : 0.80, 
        "FP" : 0.43,
        "Lioness" : 0.70
    },
    "ImageNet" : {
        "SO" : 0.91,
        "NAS" : 0.25,
        "Px" : 0.56,
        "PR" : 0.88,
        "DL-SIM" : 0.95,
        "BBBC026" : 0.90,
        "BBBC052" : 0.79,
        "BBBC053" : 0.80,        
        "F-Actin" : 0.56,
        "SPZ" : 0.65, 
        "FP" : 0.28,
        "Lioness" : 0.65
    }
}

def main():
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure", type=str, default="radial-profile", choices=["radial-profile", "fractal-dimension"])
    args = parser.parse_args()

    if args.measure == "radial-profile":
        distances = numpy.load("results/radial_profile_distances.npz")["distances"]
        names = numpy.load("results/radial_profile_distances.npz")["names"].tolist()
    elif args.measure == "fractal-dimension":
        distances = numpy.load("results/fractal_dimension_distances.npz")["distances"]
        names = numpy.load("results/fractal_dimension_distances.npz")["names"].tolist()
    else:
        raise NotImplementedError(f"`{args.measure}` is not a valid option.")

    all_xs = []
    all_ys = []
    all_pretrainings = []
    for dataset in ["SO", "NAS", "Px", "PR", "DL-SIM", "BBBC026", "BBBC052", "BBBC053"]:

        fig, ax = pyplot.subplots(figsize=(3,3))
        xy = []
        pretrainings = ["STED", "SIM", "HPA", "JUMP", "ImageNet"]
        for pretraining in pretrainings:
            idx = names.index(pretraining)
            dataset_idx = names.index(dataset)
            distance = distances[idx, dataset_idx]
            score = scores[pretraining][dataset]

            # ax.scatter(distance, score, c=COLORS[pretraining])

            xy.append((distance, score))

        xs = numpy.array([x for x, y in xy])
        ys = numpy.array([y for x, y in xy])

        # xs = (xs - xs.mean()) / xs.std()
        # ys = (ys - ys.mean()) / ys.std()
        # ys = ys / ys.max()

        all_xs.extend(xs)
        all_ys.extend(ys)
        all_pretrainings.extend(pretrainings)

        for x, y, pretraining in zip(xs, ys, pretrainings):
            ax.plot(x, y, 'o', c=COLORS[pretraining])

        # polyfit = numpy.polyfit(xs, ys, deg=1)
        # xfit = numpy.linspace(0, 0.25, 100)
        # yfit = numpy.polyval(polyfit, xfit)
        # ax.plot(xfit, yfit, color='silver', linestyle='--')
        # pearson_corr = pearsonr(xs, ys)
        # ax.text(0.02, 0.02, f"$R$ = {pearson_corr.statistic:.2f}", transform=ax.transAxes)

        ax.set_xlabel("Distance")
        ax.set_ylabel("Score")
        # ax.set(ylim=(0, 1), xlim=(0, 0.25))
        # ax.set(
        #     ylim=(-3, 3),
        #     xlim=(-3, 3)
        # )
        pearson_corr = pearsonr(all_xs, all_ys)
        pearson_corr = pearsonr(xs, ys)
        print(pearson_corr.statistic,pearson_corr.pvalue)
        ax.text(0.02, 0.02, f"$R$ = {pearson_corr.statistic:.2f}", transform=ax.transAxes)
        fig.savefig(f"figures/image-similarity/distance-vs-score_classification_{dataset}_{args.measure}.pdf", dpi=300, bbox_inches='tight')
        pyplot.close(fig)

    fig, ax = pyplot.subplots(figsize=(3,3))
    for x, y, pretraining in zip(all_xs, all_ys, all_pretrainings):
        ax.plot(x, y, 'o', c=COLORS[pretraining])
    pearson_corr = pearsonr(all_xs, all_ys)
    print(pearson_corr.pvalue)
    ax.text(0.02, 0.02, f"Overall $R$ = {pearson_corr.statistic:.2f}", transform=ax.transAxes)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Score")
    # ax.set(ylim=(-3, 3), xlim=(-3, 3))
    fig.savefig(f"figures/image-similarity/distance-vs-score_classification_overall_{args.measure}.pdf", dpi=300, bbox_inches='tight')
    pyplot.close(fig)

    for dataset in ["F-Actin", "FP", "Lioness", "SPZ"]:
        fig, ax = pyplot.subplots(figsize=(3,3))
        xy = []
        for pretraining in ["STED", "SIM", "HPA", "JUMP", "ImageNet"]:
            idx = names.index(pretraining)

            dataset_idx = names.index(dataset)
            distance = distances[idx, dataset_idx]
            score = scores[pretraining][dataset]

            ax.scatter(distance, score, c=COLORS[pretraining])
            xy.append((distance, score))

        polyfit = numpy.polyfit([x for x, y in xy], [y for x, y in xy], deg=1)
        xfit = numpy.linspace(0, 0.25, 100)
        yfit = numpy.polyval(polyfit, xfit)
        ax.plot(xfit, yfit, color='silver', linestyle='--')
        pearson_corr = pearsonr([x for x, y in xy], [y for x, y in xy])
        ax.text(0.02, 0.02, f"$R$ = {pearson_corr.statistic:.2f}", transform=ax.transAxes)

        ax.set_xlabel("Distance")
        ax.set_ylabel("Score")
        # ax.set(ylim=(0, 1), xlim=(0, 0.25))
        fig.savefig(f"figures/image-similarity/distance-vs-score_segmentation_{dataset}.pdf", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()