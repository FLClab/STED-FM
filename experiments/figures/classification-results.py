import numpy as np
import matplotlib.pyplot as plt
from RESULTS import CLASSIFICATION 
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="knn")
args = parser.parse_args()

def accuracy_vs_params(data: dict) -> None:
    fig = plt.figure()
    xparams = [data[key]["parameters"] for key in data.keys()]
    y_Imnet = [data[key]["synaptic-proteins"]["ImageNet"] for key in data.keys()]
    y_JUMP = [data[key]["synaptic-proteins"]["JUMP"] for key in data.keys()]
    y_STED = [data[key]["synaptic-proteins"]["STED"] for key in data.keys()]

    plt.scatter(xparams, y_Imnet, color='white', edgecolor='tab:red', s=100, linewidths=3, label="ImageNet")
    plt.scatter(x=xparams, y=y_JUMP, color='white', edgecolor='tab:green', s=100, linewidths=3, label="JUMP")
    plt.scatter(x=xparams, y=y_STED, color='white', edgecolor='tab:blue', s=100, linewidths=3, label="STED")

    plt.xlabel("# Params (M)")
    plt.ylabel("Synaptic proteins accuracy")
    plt.title("MAE")
    plt.xscale('log')
    plt.legend()
    fig.savefig("./accuracy-vs-params.png")


def main():
    data = CLASSIFICATION[args.task]
    accuracy_vs_params(data=data)

if __name__=="__main__":
    main()