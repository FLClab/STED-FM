import numpy as np
import matplotlib.pyplot as plt
from overall_fig import RESULTS, SUPERVISED

def compare_models(results: dict = RESULTS, task: str = "optim", probe: str = "KNN"):
    models = ["RESNET18", "RESNET50", "MAE"]
    data = [RESULTS[m][task]['STED'][probe] for m in models]
    bar_colors = ['gray', 'gray', 'gray']
    bar_labels = models
    fig = plt.figure()
    plt.bar(models, data, label=bar_labels, color=bar_colors, edgecolor='black')
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title(f"{task} | {probe}")
    fig.savefig(f"{task}_{probe}_model_comparison.pdf", bbox_inches='tight', dpi=1200)
    plt.close(fig)

 
def main():
    compare_models(probe="KNN")  
    compare_models(probe='linear-probing')
    compare_models(probe='fine-tuning')

if __name__=="__main__":
    main()