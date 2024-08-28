import matplotlib.pyplot as plt
import numpy as np
from segmentation_results import RESULTS

plt.style.use('dark_background')

def make_plot(results:dict, model: str = "MAE-small", dataset: str = "F-Actin", metric: str = "AUPR_Rings"):
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
    plt.bar(new_bar_position, new_bar_height, width, label='From scratch', color='orange')
    plt.bar(ind, hpa_data, width, label='HPA', color='cornflowerblue')
    plt.bar(ind + width, imagenet_data, width, label='ImageNet', color='violet')
    plt.bar(ind + width * 2, sted_data, width, label='STED', color='yellowgreen')
    plt.xlabel('')
    print(metric.split("_"))
    metric_name = metric.split("_")[0]
    plt.ylabel(f'Average {metric_name}')
    plt.title(f'Average {metric_name} - {model} - {dataset} - {metric}')
    plt.ylim(0, 1)
    plt.xticks(np.append(new_bar_position + width / 2, ind + width / 2), ('From scratch', 'Frozen', 'Finetuned'))
    plt.tick_params(bottom = False)
    plt.legend(loc='best')
    plt.savefig(f"{model}-{metric}.png", bbox_inches='tight')

def make_plot_fewshot(model: str = "VIT_small", dataset: str = "F-Actin", probe: str = "Finetuned") -> None:
    from_scratch = [0.5432, 0.5391, 0.5498, 0.5505]
    imnet_data = [0.5635, 0.5418, 0.5830, 0.5972]
    hpa_data = [0.5854, 0.5611, 0.5931, 0.6026]
    sted_data = [0.5855, 0.5729, 0.5948, 0.6287]
    ticks = [10, 20, 50, 100]
    ticklabels = [str(item) for item in ticks]

    fig = plt.figure()
    plt.plot(ticks, from_scratch, marker='o', label='From scratch')
    plt.plot(ticks, imnet_data, marker='o', color='tab:red', label="ImageNet")
    plt.plot(ticks, hpa_data, marker='o', color='tab:green', label="HPA")
    plt.plot(ticks, sted_data, marker='o', color='tab:blue', label="STED")
    plt.xlabel("Num of samples")
    plt.ylabel("Average AUPR")
    plt.title(f"{model} | {dataset} | {probe}")
    plt.xticks(ticks=ticks, labels=ticklabels)
    plt.legend()
    fig.savefig(f"./{model}_{dataset}_{probe}_fewshot_curves.pdf", dpi=1200, bbox_inches='tight')
    plt.close(fig)


def main():
    # make_plot(results=RESULTS, model="MAE-small", dataset="F-Actin", metric="IoU_Rings")
    # make_plot(results=RESULTS, model="MAE-small", dataset="F-Actin", metric="IoU_Fibers")
    # make_plot(results=RESULTS, model="ResNet18", dataset="Footprocess", metric="AUPR_FP")
    # make_plot(results=RESULTS, model="ResNet18", dataset="Footprocess", metric="AUPR_SD")
    make_plot_fewshot()
if __name__=="__main__":
    main()