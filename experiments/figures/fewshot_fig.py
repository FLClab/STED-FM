import numpy as np
import matplotlib.pyplot as plt 
from results import FEWSHOT

def make_plot(results: dict, model: str = "MAE_SMALL", dataset: str = "synaptic-proteins", probe: str = "fine-tuning") -> None:
    imnet_data = list(results[model][dataset]["ImageNet"][probe])
    ctc_data = list(results[model][dataset]["CTC"][probe])
    sted_data = list(results[model][dataset]["STED"][probe])
    ticks = [1, 10, 25, 50, 100]
    ticklabels = [str(item) for item in ticks]

    fig = plt.figure()
    plt.plot(ticks, imnet_data, marker='o', color='tab:red', label="ImageNet")
    plt.plot(ticks, ctc_data, marker='o', color='tab:green', label="CTC")
    plt.plot(ticks, sted_data, marker='o', color='tab:blue', label="STED")
    plt.xlabel("Label %")
    plt.ylabel("Accuracy")
    plt.title(f"{model} | {dataset} | {probe}")
    plt.xticks(ticks=ticks, labels=ticklabels)
    plt.legend()
    fig.savefig(f"./fewshot/{model}_{dataset}_{probe}_fewshot_curves.pdf", dpi=1200, bbox_inches='tight')
    plt.close(fig)

def main():
    make_plot(results=FEWSHOT, model="MAE_SMALL", dataset="synaptic-proteins", probe='linear-probing')
    make_plot(results=FEWSHOT, model="MAE_SMALL", dataset="optim", probe='linear-probing')

if __name__=="__main__":
    main()