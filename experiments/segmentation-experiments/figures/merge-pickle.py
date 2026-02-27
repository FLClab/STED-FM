
import pandas
import os 
import glob
import argparse
import pickle
import numpy

from stedfm.DEFAULTS import DATASETS
from stedfm.stats import resampling_stats

DATASETS.factin = "F-Actin"
DATASETS.synaptic_semantic_segmentation = "SPZ"
DATASETS.footprocess = "FP"
DATASETS.lioness = "Lioness"
DATASETS.lcn = "LCN"
DATASETS.deepd3 = "DeepD3"

MODEL = "mae-lightning-small"
datasets = [
    "factin",
    "synaptic-semantic-segmentation",
    "footprocess",
    "lioness",
    "lcn",
    "deepd3"
]
modes = [
    "pretrained",
    "pretrained-frozen",
]
models = [
    "MAE_SMALL_STED",
    "MAE_SMALL_SIM",
    "MAE_SMALL_HPA",
    "MAE_SMALL_JUMP",
    "MAE_SMALL_IMAGENET1K_V1",
    "from-scratch"
]

MODES = {
    "pretrained" : "FT",
    "pretrained-frozen" : "LP",
}

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default="f1", choices=["f1", "iou", "precision", "recall"])
    args = parser.parse_args()

    with pandas.ExcelWriter(os.path.join(".", "results", "raw-outputs", f"segmentation-experiments-{args.metric}-merged.xlsx")) as writer:
        for dataset in datasets:
            for mode in modes:
                samples_per_model = {}
                for model in models:
                    if mode == "pretrained-frozen" and model == "from-scratch":
                        continue
                    filepath = os.path.join("./results_v2", f"{dataset}-{mode}-f1-raw-{model}.pkl")
                    if os.path.isfile(filepath):
                        with open(filepath, "rb") as handle:
                            data = pickle.load(handle)
                        
                        samples = [
                            numpy.mean(values) for key, values in data[args.metric].items()
                        ]
                        samples_per_model[DATASETS[model]] = samples
                    else:
                        print(f"File not found: `{filepath}`")

                if len(samples_per_model) == 0:
                    print(f"No data found for dataset: `{dataset}` and mode: `{mode}`")
                    continue
                df = pandas.DataFrame(samples_per_model.values(), index=samples_per_model.keys())
                df.to_excel(writer, sheet_name=f"{DATASETS[dataset]}-{MODES[mode]}")

                samples = list(samples_per_model.values())
                labels = list(samples_per_model.keys())

                from scipy.stats import kruskal
                H_stat, H_p_value = kruskal(*samples,)
                print(f"Kruskal-Wallis H-statistic: {H_stat}, p-value: {H_p_value}")
                p_values = numpy.ones((len(samples), len(samples)))
                if H_p_value < 0.05:
                    print("  Significant differences found among groups.")
                    for i in range(len(samples)):
                        for j in range(i + 1, len(samples)):
                            from scipy.stats import ks_2samp, mannwhitneyu
                            stat, p = mannwhitneyu(samples[i], samples[j], alternative='two-sided')
                            # print(f"  Mann-Whitney U test between {labels[i]} and {labels[j]}: U-statistic={stat}, p-value={p}")
                            p_values[i, j] = p
                            p_values[j, i] = p
                    print("Pairwise Mann-Whitney U test p-values:")
                    print(p_values)
                    print("\n")
                else:
                    print("  No significant differences found among groups.")                
                p_values = pandas.DataFrame(p_values, index=labels, columns=labels)

                # print("Computing statistics for dataset:", dataset, "mode:", mode)
                # p_values, F_p_values = resampling_stats(list(samples_per_model.values()), labels=list(samples_per_model.keys()))
                p_values.to_excel(writer, sheet_name=f"{DATASETS[dataset]}-{MODES[mode]} (p={H_p_value:0.4e})")
                
if __name__ == "__main__":

    main()