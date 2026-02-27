import numpy as np 
import matplotlib.pyplot as plt 
import os
import glob 
import json 
import argparse 
from tqdm import tqdm
from matplotlib import patches
import sys 


from stedfm.DEFAULTS import BASE_PATH, COLORS, MARKERS 
from stedfm.utils import savefig 
from stedfm.stats import resampling_stats, plot_p_values

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--mode", type=str, default="pretrained-frozen", choices=["pretrained-frozen", "pretrained"],
                    help="Number of samples to plot")      
parser.add_argument("--domain", type=str, default="STED", choices=["STED", "MIC"])
parser.add_argument("--metric", type=str, default="aupr")
parser.add_argument("--sampling-mode", type=str, default="labels")
args = parser.parse_args()

print(args)

def get_metric_from_dict(data_dict, metric):
    if metric not in data_dict:
        # Maybe it's nested in other_metrics
        if "other_metrics" in data_dict and metric in data_dict["other_metrics"]:
            return data_dict["other_metrics"][metric]
        else:
            raise ValueError(f"Metric `{metric}` not found in data dictionary.")
    return data_dict[metric]

def load_file(file):
    with open(file, "r") as handle:
        data = json.load(handle)
    return data

def get_data(pretraining="STED", dataset="factin", sample="10", mode=args.mode):

    if args.mode == "from-scratch" or pretraining == "from-scratch":
        files = glob.glob(os.path.join(BASE_PATH, "segmentation-baselines", f"{args.model}", dataset, f"from-scratch*-{sample}%-labels*", f"segmentation-scores.json"), recursive=True)
    else:
        if args.sampling_mode == "samples":
            files = glob.glob(os.path.join(BASE_PATH, "segmentation-baselines", f"{args.model}", dataset, f"{mode}*_{pretraining.upper()}*-{sample}-samples*", f"segmentation-scores.json"), recursive=True)
        else:
            files = glob.glob(os.path.join(BASE_PATH, "segmentation-baselines", f"{args.model}", dataset, f"{mode}*_{pretraining.upper()}*-{sample}%-labels*", f"segmentation-scores.json"), recursive=True)

    if mode == "pretrained":
        # remove files that contains samples
        files = list(filter(lambda x: "frozen" not in x, files))  
    # remove files that contains samples
    if len(files) < 1: 
        print(f"Could not find files for mode: `{mode}`, sample `{sample}` and pretraining: `{pretraining}` ({len(files)}/5)")
        return []
    if len(files) != 5:
        print(f"Could not find all files for mode: `{mode}`, sample `{sample}` and pretraining: `{pretraining}` ({len(files)}/5)")

        required_seeds = set([42, 43, 44, 45, 46])
        found_seeds = set()
        for file in files:
            basename = os.path.dirname(file)
            found_seeds.add(int(basename.split("-")[-1]))
        missing_seeds = required_seeds - found_seeds
        print(f"  Missing seeds: {missing_seeds}")

    scores = []
    for file in files:
        scores.append(load_file(file))
    return scores

def get_full_data(mode=args.mode, pretraining="STED", dataset="factin"):
    data = {}
    if mode == "from-scratch" or pretraining == "from-scratch":
        files = glob.glob(os.path.join(BASE_PATH, "segmentation-baselines", f"{args.model}", dataset, f"from-scratch*", f"segmentation-scores.json"), recursive=True)
    else:
        files = glob.glob(os.path.join(BASE_PATH, "segmentation-baselines", f"{args.model}", dataset, f"{mode}*_{pretraining.upper()}*", f"segmentation-scores.json"), recursive=True)
        files = [f for f in files if "labels" not in f]
        if mode == "pretrained":
            files = [f for f in files if "frozen" not in f]

    if mode == "pretrained":
        # remove files that contains samples
        files = list(filter(lambda x: "frozen" not in x, files))    
        
    # remove files that contains samples
    files = list(filter(lambda x: "samples" not in x, files))
    files = list(filter(lambda x: "labels" not in x, files))
    if len(files) < 1: 
        print(f"Could not find files for full data mode: `{mode}` and pretraining: `{pretraining}` ({len(files)}/5)")
        return data
    if len(files) != 5:
        print(f"Could not find all files for full data mode: `{mode}` and pretraining: `{pretraining}` ({len(files)}/5)")

        required_seeds = set([42, 43, 44, 45, 46])
        found_seeds = set()
        for file in files:
            basename = os.path.dirname(file)
            found_seeds.add(int(basename.split("-")[-1]))
        missing_seeds = required_seeds - found_seeds
        print(f"  Missing seeds: {missing_seeds}")
        
    scores = []
    for file in files:
        scores.append(load_file(file))
    return scores

def plot_rankings(rankings: dict):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    width = 1 / (len(rankings.keys()) + 1)

    for i, pretraining in enumerate(rankings.keys()):
        sample_scores = []
        for s, sample in enumerate(rankings[pretraining].keys()):
            position = s + i / (len(rankings.keys()) + 1)
            data = rankings[pretraining][sample] 
            err = np.array([0, np.std(data)])[..., np.newaxis]
            bplot = ax.bar(position, np.mean(data), width=width, edgecolor=COLORS[pretraining], facecolor=COLORS[pretraining], yerr=err, alpha=0.4)
            ax.scatter([position] * len(data), data, color=COLORS[pretraining])
            # bplot = ax.boxplot(data, positions=[position], showfliers=True, patch_artist=True)
            # for name, parts in bplot.items():
            #     for part in parts:
            #         if name == 'boxes':
            #             # Set the fill color with alpha
            #             part.set_facecolor(COLORS[pretraining])
            #             part.set_alpha(0.3)
            #             # Set the edge color
            #             # part.set_edgecolor("black")
            #         else:
            #             # For all other elements (whiskers, caps, medians, fliers)
            #             part.set_color(COLORS[pretraining])
            #         part.set_linewidth(1.5)

            
    ax.set(
        ylabel="$AUPR^* - AUPR$",
        xticks=np.arange(5) + width * len(rankings.keys()) / 2 - 0.5 * width,
        xticklabels=["1", "10", "25", "50", "100"],
        ylim=(0, 0.25),
    )
    # ax.legend(
    #     handles=[
    #         patches.Patch(color=COLORS[label], label=label) for label in rankings.keys()
    #     ],
    #     fontsize=8
    # )
    savefig(fig, os.path.join(".", "results", f"ranking_figure_{args.model}_{args.mode}_{args.domain}"), extension="pdf")


def main():
    sample_size = ["1", "10", "25", "50", "100"]
    if args.domain.lower() == "sted":
        downstream_datasets = ["factin", "synaptic-semantic-segmentation", "footprocess", "lioness"]
    else:
        downstream_datasets = ["lcn", "deepd3"]

    downstream_datasets = [
        "factin", "synaptic-semantic-segmentation", "footprocess", "lioness", "lcn", "deepd3"
    ]

    if args.mode == "pretrained":
        pretraining_datasets = ["from-scratch", "ImageNet", "JUMP", "HPA", "SIM", "STED"]
    else:
        pretraining_datasets = ["ImageNet", "JUMP", "HPA", "SIM", "STED"]

    rankings = {
        pretraining: {
            num_samples: [] for num_samples in sample_size
        } for pretraining in pretraining_datasets
    }
    
    for i, sample in tqdm(enumerate(sample_size), total=len(sample_size)):
        for j, downstream in enumerate(downstream_datasets):
            all_scores = []
            for k, pretraining in enumerate(pretraining_datasets):
                if sample == "100":
                    scores = get_full_data(pretraining=pretraining, dataset=downstream, mode=args.mode)
                else:
                    scores = get_data(sample=sample, pretraining=pretraining, dataset=downstream, mode=args.mode)
                
                scores = [get_metric_from_dict(item, args.metric if downstream != "hpa-classification" else "f1") for item in scores]
                if len(scores) <  1:
                    continue
                all_scores.append(np.mean(scores))

            if len(all_scores) < 1:
                continue

            print(all_scores)
            max_score = max(all_scores)
            delta_scores = [max_score - score for score in all_scores]
            print(f"--- {sample} ; {downstream} ---")
            print(all_scores)
            print(delta_scores)
            print("-----------------\n")
            for p in range(len(pretraining_datasets)):
                rankings[pretraining_datasets[p]][sample].append(delta_scores[p])
    
    from scipy.stats import kruskal

    for sample in sample_size:
        print(f"--- Sample size: {sample} ---")
        samples = []
        labels = []
        for pretraining in rankings.keys():
            data = rankings[pretraining][sample]
            samples.append(data)
            labels.append(f"{pretraining}-{sample}")

        H_stat, H_p_value = kruskal(*samples,)
        print(f"Kruskal-Wallis H-statistic: {H_stat}, p-value: {H_p_value}")
        p_values = np.ones((len(samples), len(samples)))  # Initialize p-values matrix
        if H_p_value < 0.05:
            print("  Significant differences found among groups.")
            for i in range(len(samples)):
                for j in range(i + 1, len(samples)):
                    from scipy.stats import ks_2samp, mannwhitneyu, wilcoxon
                    # stat, p = ks_2samp(samples[i], samples[j])
                    # stat, p = wilcoxon(samples[i], samples[j], alternative='greater')
                    stat, p = mannwhitneyu(samples[i], samples[j], alternative='greater')
                    # print(f"  Mann-Whitney U test between {labels[i]} and {labels[j]}: U-statistic={stat}, p-value={p}")
                    p_values[i, j] = p
                    p_values[j, i] = p
        else:
            print("  No significant differences found among groups.")

        import pandas
        p_values = pandas.DataFrame(p_values, index=labels, columns=labels)
        print("Pairwise Mann-Whitney U test p-values:")
        print(p_values)
        print("\n")
        p_values.to_csv(os.path.join(".", "results", "raw-outputs", f"{args.model}_{args.domain}_{args.mode}_H{H_p_value:0.4e}_ranking_stats_sample-{sample}.csv"))

        # p_values, F_p_values = resampling_stats(samples, labels, sampling_func=np.median)
        # print(p_values)
        # print(F_p_values)

    plot_rankings(rankings)

if __name__=="__main__":
    main()