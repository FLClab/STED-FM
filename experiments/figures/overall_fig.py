"""
Simple script to visualize results from the Notion tables for the different experiments
Results are transcribed as hard-coded global variables here
"""
import numpy as np 
import matplotlib.pyplot as plt

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
    fig.savefig(f"./overall/{model}_{task}.pdf", dpi=1200)
    plt.close(fig)

def main():
    # make_plot(results=RESULTS, model="RESNET18", task='synaptic-proteins')
    make_plot(results=RESULTS, model="RESNET50", task='optim')

if __name__=="__main__":
    main()