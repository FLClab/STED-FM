
import numpy
import os, glob
import json
import argparse
import sys
import pandas
from scipy import stats
from matplotlib import pyplot

from stedfm.DEFAULTS import BASE_PATH, COLORS, MARKERS
from stedfm.utils import savefig
from stedfm.stats import resampling_stats, plot_p_values

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mae-small",
                    help="Name of the model")
parser.add_argument("--dataset", type=str, 
                    help="Name of the dataset")
parser.add_argument("--best-model", type=str, default=None, 
                    help="Which model to keep")
parser.add_argument("--metric", default="acc", type=str,
                    help="Name of the metric to access from the saved file")
parser.add_argument("--samples", nargs="+", type=str, default=None,
                    help="Number of samples to plot")
parser.add_argument("--mode", type=str, default="linear-probe", choices=["linear-probe", "finetuned"])
args = parser.parse_args()

print(args)

def load_file(file):
    with open(file, "r") as handle:
        data = json.load(handle)
    return data

def get_metric_from_dict(data_dict, metric):
    if metric not in data_dict:
        # Maybe it's nested in other_metrics
        if "other_metrics" in data_dict and metric in data_dict["other_metrics"]:
            return data_dict["other_metrics"][metric]
        else:
            raise ValueError(f"Metric `{metric}` not found in data dictionary.")
    return data_dict[metric]

def get_data(pretraining="STED"):
    data = {}
    for sample in args.samples:
        if pretraining == "from-scratch":
            files = glob.glob(os.path.join(BASE_PATH, "baselines", f"{args.model}_from-scratch", args.dataset, f"accuracy_from-scratch_{sample}_*.json"), recursive=True)
        else:
            files = glob.glob(os.path.join(BASE_PATH, "baselines", f"{args.model}_{pretraining}", args.dataset, f"accuracy_{args.mode}_{sample}_*.json"), recursive=True)
        
        if len(files) < 1:
            print(f"Could not find files for sample: `{sample}` and pretraining: `{pretraining}`")
            continue
        if len(files) != 5:
            print(f"Could not find all files for sample: `{sample}` and pretraining: `{pretraining} ({len(files)}/5)")

            required_seeds = set([42, 43, 44, 45, 46])
            found_seeds = set()
            for file in files:
                basename = os.path.dirname(file)
                found_seeds.add(int(basename.split("-")[-1]))
            missing_seeds = required_seeds - found_seeds
            print(f"  Missing seeds: {missing_seeds}")

        if len(files) > 5:
            print(f"Found more than 5 files for sample: `{sample}` and pretraining: `{pretraining} ({len(files)}/5)")
            exit()
        scores = []
        for file in files:
            scores.append(load_file(file))
        data[sample] = scores
    return data

def get_full_data(mode=args.mode, pretraining="STED"):
    data = {}
    if pretraining == "from-scratch":
        files = glob.glob(os.path.join(BASE_PATH, "baselines", f"{args.model}_from-scratch", args.dataset, f"accuracy_from-scratch_None_*.json"), recursive=True)
    else:
        files = glob.glob(os.path.join(BASE_PATH, "baselines", f"{args.model}_{pretraining}", args.dataset, f"accuracy_{mode}_None_*.json"), recursive=True)
    if len(files) < 1: 
        print(f"Could not find files for full data mode: `{mode}` and pretraining: `{pretraining}`")
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
    data[mode] = scores
    return data

def plot_data(pretraining, data, figax=None):
    full_dataset_results = get_full_data(pretraining=pretraining)
    full_dataset_results = [get_metric_from_dict(item, args.metric) for item in full_dataset_results[args.mode]]
    full_mean = numpy.mean(full_dataset_results)
    full_sem = stats.sem(full_dataset_results)

    if figax is None:
        fig, ax = pyplot.subplots()
    else:
        fig, ax = figax

    averaged = []
    errs = []
    for key, values in data.items():
        values = [get_metric_from_dict(value, args.metric) for value in values]
        mean, std = numpy.mean(values), numpy.std(values)
        sem = stats.sem(values)
        errs.append(sem)
        averaged.append(mean)
        # ax.errorbar(float(key), mean, std, color=COLORS[pretraining])
        # ax.plot(float(key), mean, color=COLORS[pretraining], marker=MARKERS[pretraining])

    averaged.append(full_mean)
    errs.append(full_sem)
    averaged = numpy.array(averaged)
    errs = numpy.array(errs)
   
    ax.plot([float(key) for key in data.keys()] + [200], averaged, color=COLORS[pretraining], marker=MARKERS[pretraining], label=pretraining)
    ax.fill_between([float(key) for key in data.keys()] + [200], averaged - errs, averaged + errs, color=COLORS[pretraining], alpha=0.2)
    ax.set(
        xlabel="Num. samples per class", ylabel=args.metric,
        # ylim=(0, 1),
        xticks=[int(s) for s in data.keys()] + [200],
        xticklabels=list(data.keys()) + ["Full"]
    )

    return (fig, ax)

def main():

    fig, ax = pyplot.subplots(figsize=(4,3))

    pretrainings = ["STED", "SIM", "HPA", "JUMP", "ImageNet"]
    if args.mode in ["finetuned"]:
        pretrainings.append("from-scratch")

    for pretraining in pretrainings:
        data = get_data(pretraining=pretraining)
        fig, ax = plot_data(pretraining, data, figax=(fig, ax))
    ax.legend()
    savefig(fig, os.path.join(".", "results", f"{args.model}_{args.dataset}_{args.mode}-small-dataset-samples"), extension="pdf")

    # Statistics
    samples, labels = [], []
    for pretraining in pretrainings:
        data = get_data(pretraining=pretraining)
        data = {key: [get_metric_from_dict(item, args.metric) for item in values] for key, values in data.items()}
        samples.extend(data.values())
        labels.extend([f"{pretraining}-{key}" for key in data.keys()])

        full_dataset_results = get_full_data(pretraining=pretraining)
        full_dataset_results = [get_metric_from_dict(item, args.metric) for item in full_dataset_results[args.mode]]
        samples.append(full_dataset_results)
        labels.append(f"{pretraining}-Full")
    
    df = pandas.DataFrame(samples, index=labels)
    os.makedirs(os.path.join(".", "results", "raw-outputs"), exist_ok=True)
    df.to_csv(os.path.join(".", "results", "raw-outputs", f"{args.model}_{args.dataset}_{args.mode}-small-dataset.csv"))

    p_values, F_p_value = resampling_stats(samples, labels)
    print("ANOVA p-value:", F_p_value)
    p_values.to_csv(os.path.join(".", "results", "raw-outputs", f"{args.model}_{args.dataset}_{args.mode}_F{F_p_value:0.4f}-small-dataset-stats.csv"))
    fig, ax = plot_p_values(p_values)
    savefig(fig, os.path.join(".", "results", f"{args.model}_{args.dataset}_{args.mode}-small-dataset-stats"), extension="pdf")

if __name__ == "__main__":
    main()