"""
Simple script to visualize results from the Notion tables for the different experiments
Results are transcribed as hard-coded global variables here
"""
import numpy as np 
import matplotlib.pyplot as plt

SUPERVISED = 0.742

RESULTS = {
    "MAE": 
        {   
            "synaptic-proteins":
                {
                    "ImageNet": {
                        "KNN": 0.397,
                        "linear-probing": 0.0,
                        "fine-tuning": 0.0,
                    },
                    "CTC": {
                        "KNN": 0.409,
                        "linear-probing": 0.0,
                        "fine-tuning": 0.0,
                    },
                    "STED": {
                        "KNN": 0.749,
                        "linear-probing": 0.0,
                        "fine-tuning": 0.0,
                    }
                },
        "optim": 
            {
                "ImageNet": {
                        "KNN": 0.797,
                        "linear-probing": 0.0,
                        "fine-tuning": 0.0,
                    },
                    "CTC": {
                        "KNN": 0.860,
                        "linear-probing": 0.0,
                        "fine-tuning": 0.0,
                    },
                    "STED": {
                        "KNN": 0.954,
                        "linear-probing": 0.0,
                        "fine-tuning": 0.0,
                    }
            }
    }
}

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
    if task == 'synaptic-proteins':
        plt.axhline(y=SUPERVISED, xmin=0, xmax=x3[-1]+w, color='grey', ls='--', label='Fully supervised')
    plt.xticks([item+0.15 for item in x1], keys)
    plt.ylabel("Accuracy")
    plt.xlabel("Classification method")
    plt.ylim([0.0, 1.0])
    plt.legend()
    plt.title(f"{model} on {task}")
    fig.savefig(f"./{model}_{task}.pdf", dpi=1200)
    plt.close(fig)

def main():
    make_plot(results=RESULTS, model="MAE", task='synaptic-proteins')
    make_plot(results=RESULTS, model="MAE", task='optim')

if __name__=="__main__":
    main()