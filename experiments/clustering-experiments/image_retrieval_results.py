import numpy as np 
import matplotlib.pyplot as plt  
from stedfm.DEFAULTS import COLORS
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="delta", choices=["delta", "raw"])
parser.add_argument("--metric", type=str, default="auc", choices=["auc", "aupr"])
args = parser.parse_args()

if __name__=="__main__":
    pretraining_datasets = ["STED", "SIM", "HPA", "JUMP", "ImageNet"]
    data = np.load(f"./results/{args.metric}_image_retrieval_results.npz")
    performance_heatmap = data["performance_heatmap"]
    max_per_dataset = np.max(performance_heatmap, axis=0)

    delta_results = np.zeros_like(performance_heatmap)
    for i in range(performance_heatmap.shape[0]):
        row = performance_heatmap[i, :]
        for j in range(performance_heatmap.shape[1]):
            delta_results[i, j] = max_per_dataset[j] - row[j]

    print(delta_results)


    results = delta_results if args.mode == "delta" else performance_heatmap

    avg_scores = np.mean(results, axis=1)
    std_scores = np.std(results, axis=1)

    fig = plt.figure(figsize=(2,3))
    ax = fig.add_subplot(111)
    bars = ax.barh(y=np.arange(len(avg_scores)), width=avg_scores)
    for bar, pretraining_dataset in zip(bars, pretraining_datasets):
        bar.set_color(COLORS[pretraining_dataset])
        bar.set_edgecolor("black")
        bar.set_alpha(0.4)

    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            ax.scatter([results[i, j]], [i], color=COLORS[pretraining_datasets[i]], edgecolor="black", marker="o")

    ax.set_xlabel(f"{args.metric}* - {args.metric}")
    ax.set_ylabel("Pretraining dataset")
    ax.set_yticks(np.arange(len(avg_scores)))
    if args.mode == "raw":
        ax.set_xlim([0.6, 1.0])
    fig.savefig(f"./results/{args.metric}_{args.mode}_image_retrieval.pdf", bbox_inches='tight', dpi=1200)
    plt.close(fig)