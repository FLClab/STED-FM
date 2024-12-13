
import numpy
import os, glob
import json
import argparse
import sys
import pandas

from matplotlib import pyplot, patches

sys.path.insert(0, "../../")
from DEFAULTS import BASE_PATH
from utils import savefig
from stats import resampling_stats, plot_p_values

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, 
                    help="Name of the dataset")
parser.add_argument("--best-model", type=str, default=None, 
                    help="Which model to keep")
parser.add_argument("--metric", default="acc", type=str,
                    help="Name of the metric to access from the saved file")
args = parser.parse_args()

print(args)

COLORS = {
    "STED" : "tab:blue",
    "HPA" : "tab:orange",
    "ImageNet" : "tab:red",
    "JUMP" : "tab:green",
    "SIM" : "tab:pink"
}
FORMATTED = {
    "ImageNet": "Image-Net",
    "JUMP": "JUMP",
    "HPA": "HPA",
    "STED": "STED",
    "SIM": "SIM",
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "resnet101": "ResNet-101",
    "linear-probe": "Linear Probe",
    "finetuned": "Finetuned"
}

def load_file(file):
    with open(file, "r") as handle:
        data = json.load(handle)
    return data

def get_data(model, mode="linear-probe", pretraining="STED"):
    data = {
        mode: []
    }
    files = glob.glob(os.path.join(BASE_PATH, "baselines", f"{pretraining}", args.dataset, f"accuracy_{mode}_None_*.json"), recursive=True)
    if len(files) < 1: 
        print(f"Could not find files for mode: `{mode}` and pretraining: `{pretraining}`")
        return data
    if len(files) != 5:
        print(f"Could not find all files for mode: `{mode}` and pretraining: `{pretraining}`")
    for file in files:
        data[mode].append(load_file(file))
    return data

def plot_data(pretraining, data, figax=None, position=0, **kwargs):

    if figax is None:
        fig, ax = pyplot.subplots()
    else:
        fig, ax = figax

    samples = []
    for key, values in data.items():
        values = [value[args.metric] for value in values]

        mean, std = numpy.mean(values), numpy.std(values)
        samples.append(values)
        # ax.bar(position, mean, yerr=std, color=COLORS[pretraining], align="edge", **kwargs)
        ax.scatter([position] * len(values), values, color=COLORS[pretraining])
        bplot = ax.boxplot(values, positions=[position], showfliers=True, **kwargs)
        for name, parts in bplot.items():
            for part in parts:
                part.set_color(COLORS[pretraining])

    return (fig, ax), samples

def main():

    fig, ax = pyplot.subplots(figsize=(4,3))
    models = ["resnet18", "resnet50", "resnet101"]
    modes = ["linear-probe", "finetuned"]
    pretrainings = ["STED", "SIM", "HPA", "JUMP", "ImageNet"]
    WEIGHTS = {
        "STED" : "{}_SIMCLR_STED",
        "SIM" : "{}_SIMCLR_SIM",
        "HPA" : "{}_SIMCLR_HPA",
        "JUMP" : "{}_SIMCLR_JUMP",
        "ImageNet" : "{}_IMAGENET1K_V1",
    }    

    df = pandas.DataFrame(
        columns=[f"{FORMATTED[mode]};{FORMATTED[pretraining]}" for mode in modes for pretraining in pretrainings],
        index=[f"{FORMATTED[model]}"for model in models],
        dtype=str
    )
    
    width = 1/(len(pretrainings) + 1)
    for model in models:
        for j, mode in enumerate(modes):
            max_col = 0
            mean = 0
            for i, pretraining in enumerate(pretrainings):
                data = get_data(model=model, mode=mode, pretraining=WEIGHTS[pretraining].format(model.upper()))
                values = [value[args.metric] for value in data[mode]]
                if numpy.mean(values) > mean:
                    max_col = i
                    mean = numpy.mean(values)
                
                if not values:
                    values = [-1] 
                df.loc[FORMATTED[model], f"{FORMATTED[mode]};{FORMATTED[pretraining]}"] = "\\SI{{ {:0.1f} \\pm {:0.1f} }}{{}}".format(numpy.mean(values) * 100, numpy.std(values) * 100)
    
            df.loc[FORMATTED[model], f"{FORMATTED[mode]};{FORMATTED[pretrainings[max_col]]}"] = "\\textbf{" + df.loc[FORMATTED[model], f"{FORMATTED[mode]};{FORMATTED[pretrainings[max_col]]}"] + "}"

    df.columns = pandas.MultiIndex.from_tuples([col.split(";") for col in df.columns])
    df.to_latex(
        f"./results/table-linear-probe-finetuned_{args.dataset}.tex",
        column_format="l" + "c" * len(df.columns),
        multicolumn_format="c",
    )

if __name__ == "__main__":
    main()