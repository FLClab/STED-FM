import matplotlib.pyplot as plt
import numpy as np
from segmentation_results import RESULTS

plt.style.use('dark_background')

def make_plot(results:dict, model: str = "MAE-small", dataset: str = "F-Actin", metric: str = "IoU_Rings"):
    data = RESULTS[dataset]
    hpa_data = [data[f"{model}_HPA"][metric][0], data[f"{model}_HPA"][metric][1]]
    imagenet_data = [data[f"{model}_ImageNet"][metric][0], data[f"{model}_ImageNet"][metric][1]]
    sted_data = [data[f"{model}_STED"][metric][0], data[f"{model}_STED"][metric][1]]

    N = 2
    new_bar_position = 0
    new_bar_height = [data[f"{model}_from-scratch"][metric]]
    
    ind = np.arange(1, N + 1)
    width = 0.2
    plt.figure(figsize=(6, 5))
    plt.bar(new_bar_position, new_bar_height, width, label='From scratch', color='purple')
    plt.bar(ind, hpa_data, width, label='HPA')
    plt.bar(ind + width, imagenet_data, width, label='ImageNet')
    plt.bar(ind + width * 2, sted_data, width, label='STED')
    plt.xlabel('')
    print(metric.split("_"))
    metric_name = metric.split("_")[0]
    plt.ylabel(f'Average {metric_name}')
    plt.title(f'Average {metric_name} - {model} - {dataset} - {metric}')
    plt.ylim(0, 1)
    plt.xticks(np.append(new_bar_position + width / 2, ind + width / 2), ('From scratch', 'Frozen', 'Pretrained'))
    plt.tick_params(bottom = False)
    plt.legend(loc='best')
    plt.savefig(f"{model}-{metric}.png", bbox_inches='tight')

def main():
    make_plot(results=RESULTS, model="ResNet18", dataset="F-Actin", metric="IoU_Rings")
    make_plot(results=RESULTS, model="ResNet18", dataset="F-Actin", metric="IoU_Fibers")
    make_plot(results=RESULTS, model="ResNet18", dataset="F-Actin", metric="AUPR_Rings")
    make_plot(results=RESULTS, model="ResNet18", dataset="F-Actin", metric="AUPR_Fibers")
if __name__=="__main__":
    main()