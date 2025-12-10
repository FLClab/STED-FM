
import glob
import numpy
import os
from matplotlib import pyplot
from matplotlib.patches import Rectangle

from collections import defaultdict
from scipy.stats import pearsonr
from sklearn.manifold import MDS
from stedfm.DEFAULTS import COLORS, DATASETS
from stedfm.utils import savefig

DATASETS.factin = "F-Actin"
DATASETS.lioness = "Lioness"
DATASETS.peroxisome = "Px"
DATASETS.footprocess = "FP"
DATASETS.polymer_rings = "PR"
DATASETS.dl_sim = "DL-SIM"
DATASETS.neural_activity_states = "NAS"
DATASETS.optim = "SO"
DATASETS.bbbc026 = "BBBC026"
DATASETS.bbbc052 = "BBBC052"
DATASETS.bbbc053 = "BBBC053"
DATASETS.lcn = "LCN"
DATASETS.deepd3 = "DeepD3"
DATASETS.synaptic_semantic_segmentation = "SPZ"
DATASETS.hpa_classification = "HPAC"
DATASETS.ov_lqhq_mt = "OV-LQHQ-MT"

# Linear-probe scores
scores = {
    "STED" : {
        "SO" : 0.96,
        "NAS" : 0.45,
        "Px" : 0.67,
        "PR" : 0.88,
        "DL-SIM" : 0.95,
        "BBBC026" : 0.93,
        "BBBC052" : 0.85,
        "BBBC053" : 0.85,
        "HPA" : 0.0972,
        "F-Actin" : 0.65,
        "SPZ" : 0.83, 
        "FP" : 0.45,
        "Lioness" : 0.71,
        "LCN" : 0.57,
        "DeepD3" : 0.84,
    },
    "SIM" : {
        "SO" : 0.91,
        "NAS" : 0.33,
        "Px" : 0.64,
        "PR" : 0.86,
        "DL-SIM" : 0.97,
        "BBBC026" : 0.88,
        "BBBC052" : 0.89,
        "BBBC053" : 0.80,     
        "HPA" : 0.0622,   
        "F-Actin" : 0.58,
        "SPZ" : 0.80, 
        "FP" : 0.40,
        "Lioness" : 0.71,
        "LCN" : 0.51,
        "DeepD3" : 0.77,  
    },
    "HPA" : {
        "SO" : 0.93,
        "NAS" : 0.25,
        "Px" : 0.65,
        "PR" : 0.84,
        "DL-SIM" : 0.91,
        "BBBC026" : 0.92,
        "BBBC052" : 0.88,
        "BBBC053" : 0.80,
        "HPA" : 0.1125,        
        "F-Actin" : 0.61,
        "SPZ" : 0.80, 
        "FP" : 0.42,
        "Lioness" : 0.71,
        "LCN" : 0.53,
        "DeepD3" : 0.84,
    },
    "JUMP" : {
        "SO" : 0.81,
        "NAS" : 0.34,
        "Px" : 0.61,
        "PR" : 0.85,
        "DL-SIM" : 0.94,
        "BBBC026" : 0.94,
        "BBBC052" : 0.86,
        "BBBC053" : 0.81,    
        "HPA" : 0.0671,    
        "F-Actin" : 0.59,
        "SPZ" : 0.80, 
        "FP" : 0.43,
        "Lioness" : 0.70,
        "LCN" : 0.53,
        "DeepD3" : 0.79,
    },
    "ImageNet" : {
        "SO" : 0.91,
        "NAS" : 0.25,
        "Px" : 0.56,
        "PR" : 0.88,
        "DL-SIM" : 0.95,
        "BBBC026" : 0.71,
        "BBBC052" : 0.70,
        "BBBC053" : 0.57,   
        "HPA" : 0.0340,     
        "F-Actin" : 0.56,
        "SPZ" : 0.65, 
        "FP" : 0.28,
        "Lioness" : 0.65,
        "LCN" : 0.48,
        "DeepD3" : 0.72,
    }
}

USER_DEFINED_CATEGORIES = {
    "STED" : "ID-STED",
    "SO" : "ID-STED",
    "NAS" : "ID-STED",
    "F-Actin" : "ID-STED",
    "SPZ" : "ID-STED",
    "Px" : "OOD-STED",
    "PR" : "OOD-STED",
    "FP" : "OOD-STED",
    "Lioness" : "OOD-STED",
    "OV-LQHQ-MT" : "OOD-STED",
    "SIM" : "OOD-MIC",
    "HPA" : "OOD-MIC",
    "HPA-Classification" : "OOD-MIC",
    "JUMP" : "OOD-MIC",
    "DL-SIM" : "OOD-MIC",
    "BBBC026" : "OOD-MIC",
    "BBBC052" : "OOD-MIC",
    "BBBC053" : "OOD-MIC",
    "LCN" : "OOD-MIC",
    "DeepD3" : "OOD-MIC",
    "ImageNet" : "OOD-NAT"
}
COLORS.dl_sim = 'gray'
COLORS.default = 'gray'
COLORS.id_sted = '#CC503E'
COLORS.ood_sted = "#F4B8E1"
COLORS.ood_mic = "#628395"
COLORS.ood_nat = '#DBAD6A'

def plot_distance_decay(distances, names, savename='radial_profile_decay'):
    fig, ax = pyplot.subplots(figsize=(3,3))
    for pretraining in ["STED", "SIM", "HPA", "JUMP", "ImageNet"]:
        pretraining_idx = names.index(pretraining)

        dist_vectors = []
        # for downstream in ["SO", "NAS", "Px", "PR", "DL-SIM", "BBBC026", "BBBC052", "BBBC053"]:
        for downstream in ["STED", "SIM", "HPA", "JUMP", "ImageNet"]:
            if pretraining == downstream:
                continue
            downstream_idx = names.index(downstream)
            dist_vector = distances[:, pretraining_idx, downstream_idx]
            dist_vector = dist_vector / dist_vector.sum()
            dist_vectors.append(dist_vector)
        mean, std = numpy.mean(dist_vectors, axis=0), numpy.std(dist_vectors, axis=0)
        ax.plot(mean, label=pretraining, color=COLORS[pretraining], alpha=1.0)
        # ax.fill_between(numpy.arange(mean.shape[0]), mean - std, mean + std, color=COLORS[pretraining], alpha=0.3)

    ax.set_xlabel("Structure Size")
    ax.set_ylabel("Explained Error")
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(["224-25", "25-10", "10-5", "5-0"])
    ax.set(
        ylim=(0, 1)
    )
    fig.savefig(f"figures/image-similarity/distance-decay_overall_{savename}.pdf", dpi=300, bbox_inches='tight')
    pyplot.close(fig)

def plot_distance_matrices(distance_matrices, names, savename='distance_matrix'):
    if distance_matrices.ndim == 2:
        distance_matrices = distance_matrices[numpy.newaxis, ...]
    pretraining = "STED"
    for distance_matrix_idx, distances in enumerate(distance_matrices):
        sorted_names = []

        distance_per_group = defaultdict(list)
        for dataset in ["STED", "SO", "NAS", "F-Actin", "SPZ"]:
            idx = names.index(pretraining)
            dataset_idx = names.index(dataset)

            distance = distances[idx, dataset_idx]
            distance_per_group["ID-STED"].append({
                "dataset" : dataset,
                "distance" : distance
            })
        
        distances_ = distance_per_group["ID-STED"]
        local_sorted_names = numpy.argsort([d["distance"] for d in distances_])
        sorted_names.extend([distances_[i]["dataset"] for i in local_sorted_names])
        
        for dataset in ["Px", "PR", "FP", "Lioness"]:
            idx = names.index(pretraining)
            dataset_idx = names.index(dataset)

            distance = distances[idx, dataset_idx]
            distance_per_group["OOD-STED"].append({
                "dataset" : dataset,
                "distance" : distance
            })

        distances_ = distance_per_group["OOD-STED"]
        local_sorted_names = numpy.argsort([d["distance"] for d in distances_])
        sorted_names.extend([distances_[i]["dataset"] for i in local_sorted_names])

        for dataset in ["HPA", "JUMP", "SIM", "DL-SIM", "BBBC026", "BBBC052", "BBBC053", "HPA", "LCN", "DeepD3"]:
            idx = names.index(pretraining)
            dataset_idx = names.index(dataset)

            distance = distances[idx, dataset_idx]
            distance_per_group["OOD-MIC"].append({
                "dataset" : dataset,
                "distance" : distance
            })
        
        distances_ = distance_per_group["OOD-MIC"]
        local_sorted_names = numpy.argsort([d["distance"] for d in distances_])
        sorted_names.extend([distances_[i]["dataset"] for i in local_sorted_names])
        
        for dataset in ["ImageNet"]:
            idx = names.index(pretraining)
            dataset_idx = names.index(dataset)

            distance = distances[idx, dataset_idx]
            distance_per_group["OOD-NAT"].append({
                "dataset" : dataset,
                "distance" : distance
            })

        distances_ = distance_per_group["OOD-NAT"]
        local_sorted_names = numpy.argsort([d["distance"] for d in distances_])
        sorted_names.extend([distances_[i]["dataset"] for i in local_sorted_names])

        sorted_dataset_indices = [names.index(name) for name in sorted_names]
        distances = distances[numpy.ix_(sorted_dataset_indices, sorted_dataset_indices)]

        mask = numpy.triu(numpy.ones_like(distances, dtype=bool), k=1)
        distance_masked = numpy.ma.array(distances, mask=mask)

        fig, ax = pyplot.subplots(figsize=(6,6))
        im = ax.imshow(distance_masked, cmap='RdPu')

        delta = 0.5
        patch = Rectangle((0 - delta,0 - delta), len(distance_per_group["ID-STED"]), len(distance_per_group["ID-STED"]), fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(patch)

        patch = Rectangle((len(distance_per_group["ID-STED"]) - delta, len(distance_per_group["ID-STED"]) - delta), len(distance_per_group["OOD-STED"]), len(distance_per_group["OOD-STED"]), fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(patch)

        patch = Rectangle((len(distance_per_group["ID-STED"]) + len(distance_per_group["OOD-STED"]) - delta, len(distance_per_group["ID-STED"]) + len(distance_per_group["OOD-STED"]) - delta), len(distance_per_group["OOD-MIC"]), len(distance_per_group["OOD-MIC"]), fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(patch)

        patch = Rectangle((len(distance_per_group["ID-STED"]) + len(distance_per_group["OOD-STED"]) + len(distance_per_group["OOD-MIC"]) - delta, len(distance_per_group["ID-STED"]) + len(distance_per_group["OOD-STED"]) + len(distance_per_group["OOD-MIC"]) - delta), len(distance_per_group["OOD-NAT"]), len(distance_per_group["OOD-NAT"]), fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(patch)

        for name in ["STED", "SIM", "HPA", "JUMP", "ImageNet"]:
            idx = sorted_names.index(name)
            ax.annotate(distances[idx, idx].round(2), (idx, idx), fontsize=6, alpha=1.0,
                        horizontalalignment='center', verticalalignment='center', 
                        color='black')
            # ax.text(idx, -1, name, rotation=90, va='bottom', ha='center', fontsize=8, weight='bold', color=COLORS[name])
            # ax.text(-1, idx, name, va='center', ha='right', fontsize=8, weight='bold', color=COLORS[name])

        ax.set_xticks(numpy.arange(len(sorted_names)))
        ax.set_yticks(numpy.arange(len(sorted_names)))
        ax.set_xticklabels(sorted_names, rotation=45, ha='right')
        ax.set_yticklabels(sorted_names)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Distance")
        fig.savefig(f"figures/image-similarity/{savename}_{distance_matrix_idx}.pdf", dpi=300, bbox_inches='tight')
        pyplot.close(fig)

        fig, ax = pyplot.subplots(figsize=(3,3))
        for key, values in distance_per_group.items():
            datasets = [v["dataset"] for v in values]
            values = [v["distance"] for v in values]
            # ax.scatter([key] * len(values), values, color='gray', alpha=0.7)
            for dataset in datasets:
                ax.annotate(dataset, (key, values[datasets.index(dataset)]), fontsize=8, alpha=1.0, 
                            horizontalalignment='center', verticalalignment='center')
            ax.scatter(key, numpy.mean(values), color='red', s=100, marker='x')
        ax.set_ylim(
            distances[sorted_names.index(pretraining)].min() - 0.05, 
            distances[sorted_names.index(pretraining)].max() + 0.05)
        ax.set_ylabel("Distance")
        ax.set_xticks(["ID-STED", "OOD-STED", "OOD-MIC", "OOD-NAT"])
        ax.set_xticklabels(["STED", "Other STED", "OOD-MIC", "OOD-NAT"], rotation=45, ha='right')
        fig.savefig(f"figures/image-similarity/{savename}_groups_{distance_matrix_idx}.pdf", dpi=300, bbox_inches='tight')
        pyplot.close(fig)    

def plot_mds(distances, names, colors=None, savename='mds'):
    
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(distances)

    fig, ax = pyplot.subplots(figsize=(6,6))
    for idx, name in enumerate(names):
        ax.annotate(name, (coords[idx,0], coords[idx,1]), fontsize=8, alpha=1.0, 
                    horizontalalignment='center', verticalalignment='center', 
                    color=colors[idx] if colors is not None else COLORS[name], weight='bold')
    ax.set_aspect('equal')
    ax.set_ylim(coords[:,1].min() - 0.1, coords[:,1].max() + 0.1)
    ax.set_xlim(coords[:,0].min() - 0.1, coords[:,0].max() + 0.1)
    ax.set_axis_off()
    fig.savefig(f"figures/image-similarity/{savename}.pdf", dpi=300, bbox_inches='tight')
    pyplot.close(fig)

def load_files(paths):
    names, files = [], []
    for path in paths:
        names.append(DATASETS[os.path.splitext(os.path.basename(path))[0]])
        files.append(numpy.load(path))
    return names, files

def plot_perimetric_complexity(names, pcs, savename='perimetric_complexity_distribution'):
    pretrainings = ["STED", "SIM", "HPA", "JUMP", "ImageNet"]
    fig, ax = pyplot.subplots(figsize=(3,3))
    xs, ys = [], []
    for pretraining in pretrainings:
        performances = [scores[pretraining][dataset] for dataset in names]
        ax.scatter(pcs, performances, label=pretraining, color=COLORS[pretraining])

        xs.extend(pcs)
        ys.extend(performances)

    for name, pc in zip(names, pcs):
        ax.annotate(name, (pc, 0.), fontsize=8, alpha=1.0,
                    rotation=90,
                    horizontalalignment='center', verticalalignment='bottom', 
                    color='gray', weight='bold')

    pearson_corr = pearsonr(xs, ys)
    print(f"Pearson correlation: {pearson_corr}")
    ax.annotate(f"$R$ = {pearson_corr.statistic:.2f}\n$p$ = {pearson_corr.pvalue:.2e}", (0.98, 0.98), fontsize=8, alpha=1.0,
                xycoords='axes fraction',
                horizontalalignment='right', verticalalignment='top', 
                color='black')

    ax.set_xlabel("Perimetric Complexity")
    ax.set_ylabel("Performance")
    ax.set(ylim=(0, 1))
    fig.savefig(f"figures/image-similarity/{savename}.pdf", dpi=300, bbox_inches='tight')
    pyplot.close(fig)

def main():
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure", type=str, default="radial-profile", choices=["radial-profile", "fractal-dimension", "fractal-dimension-annotations", "perimetric-complexity", "kl-divergence"], help="Image similarity measure to analyze.")
    args = parser.parse_args()

    if args.measure == "radial-profile":
        distances = numpy.load("results/radial_profile_distances.npz")["distances"]
        names = numpy.load("results/radial_profile_distances.npz")["names"].tolist()
    elif args.measure == "fractal-dimension":
        distances = numpy.load("results/fractal_dimension_distances.npz")["distances"]
        names = numpy.load("results/fractal_dimension_distances.npz")["names"].tolist()
    elif args.measure == "fractal-dimension-annotations":
        pass 
        return
    elif args.measure == "kl-divergence":
        distances = numpy.load("results/radial_profile_distances_kl-divergence.npz")["distances"]
        names = numpy.load("results/radial_profile_distances_kl-divergence.npz")["names"].tolist()
    elif args.measure == "perimetric-complexity":
        names, pcs = load_files([
            "results/perimetric-complexity/factin.npy",
            "results/perimetric-complexity/lioness.npy",
            "results/perimetric-complexity/footprocess.npy",
            "results/perimetric-complexity/synaptic-semantic-segmentation.npy",
        ])
        pcs = [numpy.mean(pc) for pc in pcs]
        plot_perimetric_complexity(names, pcs, savename='perimetric_complexity_distribution')
        return
    else:
        raise NotImplementedError(f"`{args.measure}` is not a valid option.")
    
    if distances.ndim == 3:
        print("Distance matrix has 3 dimensions, plotting distance decay...")
        plot_distance_decay(distances, names, savename=f'distance_decay_{args.measure}')
        plot_distance_matrices(distances, names, savename=f'distance_matrix_{args.measure}')

        distances = distances.sum(axis=0)

    plot_distance_matrices(distances, names, savename=f'distance_matrix_{args.measure}_overall')

    plot_mds(distances, names, savename=f'mds_{args.measure}_overall')
    plot_mds(distances, names, colors=[COLORS[USER_DEFINED_CATEGORIES[name]] for name in names], savename=f'mds_{args.measure}_per-category')    

    normalization_per_dataset = {}
    for dataset in ["SO", "NAS", "Px", "PR", "DL-SIM", "BBBC026", "BBBC052", "BBBC053", "HPA", "F-Actin", "SPZ", "FP", "Lioness", "LCN", "DeepD3"]:
        xy = []
        pretrainings = ["STED", "SIM", "HPA", "JUMP", "ImageNet"]
        for pretraining in pretrainings:
            idx = names.index(pretraining)
            dataset_idx = names.index(dataset)
            distance = distances[idx, dataset_idx]
            score = scores[pretraining][dataset]

            xy.append((distance, score))

        xs = numpy.array([x for x, y in xy])
        ys = numpy.array([y for x, y in xy])
        normalization_per_dataset[dataset] = {
            "x" : (xs.min(), xs.max(), xs.mean(), xs.std()),
            "y" : (ys.min(), ys.max(), ys.mean(), ys.std())
        }   

    all_xs = []
    all_ys = []
    all_pretrainings = []
    all_datasets = []
    for dataset in ["SO", "NAS", "Px", "PR", "DL-SIM", "BBBC026", "BBBC052", "BBBC053", "HPA", "F-Actin", "SPZ", "FP", "Lioness", "LCN", "DeepD3"]:

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

        sorted_indices = numpy.argsort(ys)
        xs = xs[sorted_indices]
        ys = ys[sorted_indices]
        pretrainings = [pretrainings[i] for i in sorted_indices]
        for x, y, pretraining in zip(xs, ys, pretrainings):
            ax.plot(x, y, 'o', c=COLORS[pretraining])
        # ax.plot(xs, ys, color='gray', linestyle='--', alpha=0.5)

        xs = (xs - xs.mean()) / xs.std()
        ys = (ys - ys.mean()) / ys.std()
        # ys = ys / ys.max()
        # xs = xs / xs.max()

        all_xs.extend(xs)
        all_ys.extend(ys)
        all_pretrainings.extend(pretrainings)
        all_datasets.extend([dataset] * len(xs))

        # polyfit = numpy.polyfit(xs, ys, deg=1)
        # xfit = numpy.linspace(0, 0.25, 100)
        # yfit = numpy.polyval(polyfit, xfit)
        # ax.plot(xfit, yfit, color='silver', linestyle='--')
        # pearson_corr = pearsonr(xs, ys)
        # ax.text(0.02, 0.02, f"$R$ = {pearson_corr.statistic:.2f}", transform=ax.transAxes)

        ax.set_xlabel("Distance")
        ax.set_ylabel("Score")
        ax.set(ylim=(0, 1), xlim=(0, 1.0))
        ax.set(
            ylim=(-2.5, 2.5),
            xlim=(-2.5, 2.5)
        )
        pearson_corr = pearsonr(all_xs, all_ys)
        pearson_corr = pearsonr(xs, ys)
        print(pearson_corr.statistic,pearson_corr.pvalue)
        ax.text(0.02, 0.02, f"$R$ = {pearson_corr.statistic:.2f}", transform=ax.transAxes)
        fig.savefig(f"figures/image-similarity/distance-vs-score_classification_{dataset}_{args.measure}.pdf", dpi=300, bbox_inches='tight')
        pyplot.close(fig)

    fig, ax = pyplot.subplots(figsize=(3,3))
    for x, y, pretraining, dataset in zip(all_xs, all_ys, all_pretrainings, all_datasets):
        # ax.scatter(x, y, c=COLORS[pretraining])
        ax.annotate(dataset, (x, y), fontsize=6, alpha=1.0, 
                    horizontalalignment='center', verticalalignment='center', 
                    color=COLORS[pretraining], weight='bold')
    pearson_corr = pearsonr(all_xs, all_ys)
    print(pearson_corr.pvalue)

    ax.text(0.02, 0.02, f"Overall $R$ = {pearson_corr.statistic:.2f}", transform=ax.transAxes)
    ax.set_xlabel("Standardized Distance")
    ax.set_ylabel("Standardized Score")
    ax.set(ylim=(-2.5, 2.5), xlim=(-2.5, 2.5))
    # ax.set(ylim=(0, 1.1))
    fig.savefig(f"figures/image-similarity/distance-vs-score_classification_overall_{args.measure}.pdf", dpi=300, bbox_inches='tight')
    pyplot.close(fig)

    figaxes = {pretraining: pyplot.subplots(figsize=(3,3)) for pretraining in ["STED", "SIM", "HPA", "JUMP", "ImageNet"]}
    for x, y, pretraining, dataset in zip(all_xs, all_ys, all_pretrainings, all_datasets):
        fig, ax = figaxes[pretraining]
        ax.annotate(dataset, (x, y), fontsize=8, alpha=1.0, 
                    horizontalalignment='center', verticalalignment='center', 
                    color=COLORS[pretraining], weight='bold')
    for pretraining, (fig, ax) in figaxes.items():
        pearson_corr = pearsonr(
            [x for x, p in zip(all_xs, all_pretrainings) if p == pretraining],
            [y for y, p in zip(all_ys, all_pretrainings) if p == pretraining]
        )
        ax.axhline(0, color='silver', linestyle='--', alpha=0.5)
        ax.axvline(0, color='silver', linestyle='--', alpha=0.5)
        ax.text(0.02, 0.02, f"$R$ = {pearson_corr.statistic:.2f}", transform=ax.transAxes)
        ax.set_xlabel("Distance")
        ax.set_ylabel("Score")
        ax.set(ylim=(-2.5, 2.5), xlim=(-2.5, 2.5))
        fig.savefig(f"figures/image-similarity/distance-vs-score_classification_overall_{pretraining}_{args.measure}.pdf", dpi=300, bbox_inches='tight')
        pyplot.close(fig)

    # Resampling stats from pretraining perspectives
    all_xs = []
    all_ys = []
    all_pretrainings = []
    all_datasets = []
    for dataset in ["SO", "NAS", "Px", "PR", "DL-SIM", "BBBC026", "BBBC052", "BBBC053", "HPA"]:
        xy = []
        pretrainings = ["STED", "SIM", "HPA", "JUMP", "ImageNet"]
        for pretraining in pretrainings:
            idx = names.index(pretraining)
            dataset_idx = names.index(dataset)
            distance = distances[idx, dataset_idx]
            score = scores[pretraining][dataset]

            xy.append((distance, score))

        xs = numpy.array([x for x, y in xy])
        ys = numpy.array([y for x, y in xy])

        # xs = (xs - xs.mean()) / xs.std()
        # ys = (ys - ys.mean()) / ys.std()
        ys = ys / ys.max()
        # xs = xs / xs.max()

        all_xs.extend(xs)
        all_ys.extend(ys)
        all_pretrainings.extend(pretrainings)
        all_datasets.extend([dataset] * len(xs))

    fig, ax = pyplot.subplots(figsize=(3,3))
    for x, y, pretraining, dataset in zip(all_xs, all_ys, all_pretrainings, all_datasets):
        ax.scatter(x, y, c=COLORS[pretraining])
        # ax.annotate(dataset, (x, y), fontsize=6, alpha=1.0, 
        #             horizontalalignment='center', verticalalignment='center', 
        #             color=COLORS[pretraining], weight='bold')
    ax.set(
        xlabel="Distance",
        ylabel="Score",
        # ylim=(0.5, 1.05),
        # xlim=(0, 0.5)
    )
    
    polyfit = numpy.polyfit(all_xs, all_ys, deg=1)
    xfit = numpy.linspace(min(all_xs), max(all_xs), 100)
    yfit = numpy.polyval(polyfit, xfit)
    ax.plot(xfit, yfit, color='silver', linestyle='--')

    pearson_corr = pearsonr(all_xs, all_ys)

    print("---")
    print("Overall correlation stats (resampling):")
    print(f"Pearson correlation: R = {pearson_corr.statistic:.4f}, p-value = {pearson_corr.pvalue:.4f}")

    sampled_pearson_corr = []
    for _ in range(1000):
        choices_xs = numpy.random.choice(all_xs, size=len(all_xs), replace=True)
        choices_ys = numpy.random.choice(all_ys, size=len(all_ys), replace=True)
        sampled_pearson_corr.append(pearsonr(choices_xs, choices_ys).statistic)
    sampled_pearson_corr = numpy.array(sampled_pearson_corr)
    p_value = 1 - (sampled_pearson_corr >= pearson_corr.statistic).sum() / len(sampled_pearson_corr)
    print(f"Overall correlation p-value (resampling): {p_value:.4f}")
    ax.text(0.02, 0.02, f"Overall $R$ = {pearson_corr.statistic:.2f}", transform=ax.transAxes)
    savefig(fig, f"figures/image-similarity/distance-vs-score_classification_overall_resampling_{args.measure}")
    print("Overall correlation stats:")
    print(f"Pearson correlation: R = {pearson_corr.statistic:.4f}, p-value = {pearson_corr.pvalue:.4f}")

    for pretraining in ["STED", "SIM", "HPA", "JUMP", "ImageNet"]:
        fig, ax = pyplot.subplots(figsize=(3,3))
        xy = []
        for dataset in ["SO", "NAS", "Px", "PR", "DL-SIM", "BBBC026", "BBBC052", "BBBC053", "HPA", "F-Actin", "SPZ", "FP", "Lioness", "LCN", "DeepD3"]:
            idx = names.index(pretraining)

            dataset_idx = names.index(dataset)
            distance = distances[idx, dataset_idx]
            score = scores[pretraining][dataset]

            # distance = distance / normalization_per_dataset[dataset]["x"][1]
            score = score / normalization_per_dataset[dataset]["y"][1]

            # distance = (distance - normalization_per_dataset[dataset]["x"][0]) / (normalization_per_dataset[dataset]["x"][1] - normalization_per_dataset[dataset]["x"][0])
            # score = (score - normalization_per_dataset[dataset]["y"][0]) / (normalization_per_dataset[dataset]["y"][1] - normalization_per_dataset[dataset]["y"][0])

            # distance = (distance - normalization_per_dataset[dataset]["x"][2]) / normalization_per_dataset[dataset]["x"][3]
            # score = (score - normalization_per_dataset[dataset]["y"][2]) / normalization_per_dataset[dataset]["y"][3]

            # ax.scatter(distance, score, c=COLORS[dataset])
            xy.append((distance, score))

        xs = numpy.array([x for x, y in xy])
        ys = numpy.array([y for x, y in xy])

        # xs = (xs - xs.mean()) / xs.std()
        # ys = (ys - ys.mean()) / ys.std()
        ax.scatter(xs, ys, c=COLORS[pretraining])

        # polyfit = numpy.polyfit(xs, ys, deg=1)
        # xfit = numpy.linspace(0, 0.25, 100)
        # yfit = numpy.polyval(polyfit, xfit)
        # ax.plot(xfit, yfit, color='silver', linestyle='--')
        pearson_corr = pearsonr(xs, ys)
        print(f"({pretraining}) Pearson correlation p-value: {pearson_corr.pvalue:.4f}")
        ax.text(0.02, 0.02, f"$R$ = {pearson_corr.statistic:.2f}", transform=ax.transAxes)

        ax.set_xlabel("Distance")
        ax.set_ylabel("Score")
        ax.set(ylim=(0, 1), xlim=(0, 1))
        # ax.set(ylim=(-3, 3), xlim=(-3, 3))
        fig.savefig(f"figures/image-similarity/distance-vs-score_classification_{pretraining}_{args.measure}.pdf", dpi=300, bbox_inches='tight')

    # Plots std vs std
    fig, ax = pyplot.subplots(figsize=(3,3))
    xs, ys = [], []
    for dataset in ["SO", "NAS", "Px", "PR", "DL-SIM", "BBBC026", "BBBC052", "BBBC053", "HPA", "F-Actin", "SPZ", "FP", "Lioness", "LCN", "DeepD3"]:
        distance_std = normalization_per_dataset[dataset]["x"][3]
        score_std = normalization_per_dataset[dataset]["y"][3]
        ax.scatter(distance_std, score_std, c="k")
        xs.append(distance_std)
        ys.append(score_std)
    pearson_corr = pearsonr(xs, ys)
    print(f"(std) Pearson correlation p-value: {pearson_corr.pvalue:.4f}")
    ax.text(0.02, 0.02, f"$R$ = {pearson_corr.statistic:.2f}", transform=ax.transAxes)
    ax.set(
        # xlim=(0, 0.2),
        # ylim=(0, 0.1),
        xlabel="Distance Std Dev",
        ylabel="Score Std Dev"
    )
    savefig(fig, f"figures/image-similarity/distance-vs-score_std_correlation_{args.measure}")
    pyplot.close(fig)

    print("----")
    print("Intra-distance vs score std dev correlation:")


    # Plots dataset intra-distance vs std in scores
    fig, ax = pyplot.subplots(figsize=(3,3))
    xs = []
    ys = []
    for pretraining in ["STED", "SIM", "HPA", "JUMP", "ImageNet"]:
        xy = []
        for dataset in ["SO", "NAS", "Px", "PR", "DL-SIM", "BBBC026", "BBBC052", "BBBC053", "HPA", "F-Actin", "SPZ", "FP", "Lioness", "LCN", "DeepD3"]:
            idx = names.index(pretraining)

            dataset_idx = names.index(dataset)
            distance = distances[idx, dataset_idx]
            score = scores[pretraining][dataset]

            # score = score / normalization_per_dataset[dataset]["y"][1]
            score = (score - normalization_per_dataset[dataset]["y"][2]) / normalization_per_dataset[dataset]["y"][3]
            xy.append((distance, score))

        x = distances[idx, idx]
        # ys_ = [y for x, y in xy]
        # q1, q3 = numpy.percentile(ys_, [25, 75])
        # iqr = q3 - q1
        # lower_bound = q1 - 1.5 * iqr
        # upper_bound = q3 + 1.5 * iqr
        # ys_ = [y for y in ys_ if (y >= lower_bound) and (y <= upper_bound)]
        # y = numpy.std([y for x, y in xy])
        # xs.append(x)
        # ys.append(y)

        xs.extend([x] * len(xy))
        ys.extend([y for x, y in xy])

        x = [x] * len(xy)
        y = [y for x, y in xy]
        ax.scatter(x, y, c=COLORS[pretraining])
    ax.set(
        xlabel="Intra-Distance",
        ylabel="Standardized Score",
        ylim=(-2.5, 2.5)
    )
    
    # Linear fit and correlation
    polyfit = numpy.polyfit(xs, ys, deg=1)
    delta = max(xs) - min(xs)
    xfit = numpy.linspace(min(xs) - delta * 0.1, max(xs) + delta * 0.1, 100)
    yfit = numpy.polyval(polyfit, xfit)
    ax.plot(xfit, yfit, color='silver', linestyle='--')

    pearson_corr = pearsonr(xs, ys)
    print(f"(intra-distance vs standardized score) Pearson correlation: {pearson_corr.statistic:.4f}")
    print(f"(intra-distance vs standardized score) Pearson correlation p-value: {pearson_corr.pvalue:.4f}")
    ax.text(0.02, 0.02, f"$R$ = {pearson_corr.statistic:.2f}", transform=ax.transAxes)

    savefig(fig, f"figures/image-similarity/intra-distance-vs-score-std_correlation_{args.measure}")
    pyplot.close(fig)

    print("----")
    print("Intra-distance vs score std dev correlation:")

    # Lets perform a statistical test to ensure that the correlations are significant
    samples = []
    numpy.random.seed(42)
    for _ in range(10000):
        # This shuffles the ys values to create a null distribution
        choices_ys = numpy.random.choice(ys, size=len(ys), replace=False)
        sampled_pearson_corr = pearsonr(xs, choices_ys)
        samples.append(sampled_pearson_corr.statistic)
    samples = numpy.array(samples)
    print(f"Pearson correlation for intra-distance vs score std: {numpy.mean(samples):.4f} ± {numpy.std(samples):.4f}")
    print(f"p-value for intra-distance vs score std correlation: {1 - (samples <= pearson_corr.statistic).sum() / len(samples):.4f}")

    fig, ax = pyplot.subplots(figsize=(3,3))
    ax.hist(samples, bins=20, color='gray', alpha=0.7)
    ax.axvline(pearson_corr.statistic, color='mediumpurple', linestyle='--')
    ax.set(
        xlabel="Pearson Correlation",
        ylabel="Frequency"
    )
    savefig(fig, f"figures/image-similarity/intra-distance-vs-score-std_correlation_stats_{args.measure}")

    # Plots for segmentation datasets
    # for dataset in ["F-Actin", "FP", "Lioness", "SPZ"]:
    #     fig, ax = pyplot.subplots(figsize=(3,3))
    #     xy = []
    #     for pretraining in ["STED", "SIM", "HPA", "JUMP", "ImageNet"]:
    #         idx = names.index(pretraining)

    #         dataset_idx = names.index(dataset)
    #         distance = distances[idx, dataset_idx]
    #         score = scores[pretraining][dataset]

    #         ax.scatter(distance, score, c=COLORS[pretraining])
    #         xy.append((distance, score))

    #     polyfit = numpy.polyfit([x for x, y in xy], [y for x, y in xy], deg=1)
    #     xfit = numpy.linspace(0, 0.25, 100)
    #     yfit = numpy.polyval(polyfit, xfit)
    #     ax.plot(xfit, yfit, color='silver', linestyle='--')
    #     pearson_corr = pearsonr([x for x, y in xy], [y for x, y in xy])
    #     ax.text(0.02, 0.02, f"$R$ = {pearson_corr.statistic:.2f}", transform=ax.transAxes)

    #     ax.set_xlabel("Distance")
    #     ax.set_ylabel("Score")
    #     # ax.set(ylim=(0, 1), xlim=(0, 0.25))
    #     fig.savefig(f"figures/image-similarity/distance-vs-score_segmentation_{dataset}.pdf", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()