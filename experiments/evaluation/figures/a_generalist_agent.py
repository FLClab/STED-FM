import numpy as np 
import matplotlib.pyplot as plt 
import sys 
sys.path.insert(0, "../../")
from DEFAULTS import COLORS 

RESULTS = {
    "knn": {
        "ImageNet": 1,
        "JUMP": 0,
        "HPA": 0,
        "SIM": 0,
        "STED": 4,
    },
    "classification": {
        "ImageNet": 6,
        "JUMP": 3,
        "HPA": 4,
        "SIM": 7,
        "STED": 30,
    },
    "segmentation": {
        "ImageNet": 1,
        "JUMP": 1,
        "HPA": 0,
        "SIM": 1,
        "STED": 16,
    },
    "total": {
        "ImageNet": 8,
        "JUMP": 4,
        "HPA": 4,
        "SIM": 8,
        "STED": 50,
    }
}

def plot_results(results: dict) -> None:
    for key in results.keys():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        data = [item for item in results[key].values()]
        data = [item / sum(data) for item in data]
        pretrainings = list(results[key].keys())
        positions = np.arange(len(pretrainings))
        for pos, p, d in zip(positions, pretrainings, data):
            ax.bar(pos, d, color=COLORS[p])
        ax.set_ylabel("Proportion best performing model")
        ax.set_xlabel("Pretraining dataset")
        ax.set_xticks(positions)
        ax.set_xticklabels(pretrainings)
        fig.savefig(f"../results/generalist_agent/{key}.pdf", dpi=1200, bbox_inches="tight", transparent=True)
        plt.close(fig)

if __name__=="__main__":
    plot_results(RESULTS)




