"""
Simple script to visualize results from the Notion tables for the different experiments
Results are transcribed as hard-coded global variables here
"""
import numpy as np 
import matplotlib.pyplot as plt
from experiments.figures.old_results import RESULTS, SUPERVISED
plt.style.use("dark_background")

def temporary_plot(results: dict, model: str, task: str) -> None:
    data = [
        list(results[model][task]["ImageNet"].values())[0],
        list(results[model][task]["CTC"].values())[0],
        list(results[model][task]["STED"].values())[0],
        ]
    w = 0.5
    x1 = np.arange(0, len(data), 1)
    fig = plt.figure()
    plt.bar(x1, data, color='gainsboro', width=w, edgecolor='white')
    if SUPERVISED[model][task] is not None:
        plt.axhline(y=SUPERVISED[model][task], xmin=0, xmax=x1[-1]+w, color='white', ls='--', label='Fully supervised')
    plt.xticks(x1, ['ImageNet', "CTC", "STED"])
    plt.ylabel("Accuracy")
    plt.xlabel("Pretraining data")
    plt.ylim([0.0, 1.0])
    plt.legend()
    plt.title(f"{model} KNN classification on {task}")
    fig.savefig(f"./overall/only_KNN_{model}_{task}.pdf", dpi=1200)
    plt.close(fig)

def make_plot(results: dict, model: str = "MAE", task: str = "synaptic-proteins") -> None:
    imnet_data = list(results[model][task]["ImageNet"].values())
    ctc_data = list(results[model][task]["CTC"].values())
    sted_data = list(results[model][task]["STED"].values())
    imnet_keys = list(results[model][task]["ImageNet"].keys())
    sted_keys = list(results[model][task]["STED"].keys())
    assert imnet_keys == sted_keys

    keys = sted_keys
    w = 0.2
    x1 = np.arange(0, len(imnet_data), 1)
    x2 = [item + w for item in x1]
    x3 = [item + w for item in x2]
    fig = plt.figure()
    plt.bar(x1, imnet_data, label="ImageNet", color='tab:red', width=w, edgecolor='black')
    plt.bar(x2, ctc_data, label='CTC', color='tab:green', width=w, edgecolor='black')
    plt.bar(x3, sted_data, label="STED", color="tab:blue", width=w, edgecolor='black')
    if SUPERVISED[model][task] is not None:
        plt.axhline(y=SUPERVISED[model][task], xmin=0, xmax=x3[-1]+w, color='grey', ls='--', label='Fully supervised')
    plt.xticks([item+0.15 for item in x1], keys)
    plt.ylabel("Accuracy")
    plt.xlabel("Classification method")
    plt.ylim([0.0, 1.0])
    plt.legend(loc='lower left')
    plt.title(f"{model} on {task}")
    fig.savefig(f"./overall/only_KNN_{model}_{task}.pdf", dpi=1200)
    plt.close(fig)

def main():
    temporary_plot(results=RESULTS, model="MAE_SMALL", task="synaptic-proteins")
    # make_plot(results=RESULTS, model="MAE_SMALL", task='synaptic-proteins')

if __name__=="__main__":
    main()