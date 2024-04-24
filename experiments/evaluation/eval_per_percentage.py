import torch
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import argparse 
import sys
sys.path.insert(0, "../")
from model_builder import get_classifier_v2 
from loaders import get_dataset 
from utils import compute_Nary_accuracy

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='mae')
parser.add_argument("--probe-type", type=str, default='finetuning')
parser.add_argument("--dataset", type=str, default='synaptic-proteins')
args = parser.parse_args()

def evaluate(
        model,
        loader, 
        device
):
    model.eval()
    big_correct = np.array([0] * (4+1))
    big_n = np.array([0] * (4+1))
    with torch.no_grad():
        for imgs, data_dict in tqdm(loader, desc="Evaluation..."):
            labels = data_dict['label']
            imgs, labels = imgs.to(device), labels.type(torch.LongTensor).to(device)
            predictions = model(imgs)
            correct, n = compute_Nary_accuracy(predictions, labels)
            big_correct = big_correct + correct
            big_n = big_n + n
        accuracies = big_correct / big_n
        print("Overall accuracy = {:.3f}".format(accuracies[0]))
        for i in range(1, 4+1):
            acc = accuracies[i]
            print("Class {} accuracy = {:.3f}".format(
                i, acc))
        print('\n')
    return accuracies[0]

def make_plot(accuracies: dict) -> None:
    imnet_data = list(accuracies["ImageNet"])
    ctc_data = list(accuracies["CTC"])
    sted_data = list(accuracies["STED"])

    x = np.arange(0, len(sted_data), 1)
    ticklabels = ["1", "10", "25", "50"]
    fig = plt.figure()
    plt.plot(x, imnet_data, color='tab:red', marker='x', label="ImageNet")
    plt.plot(x, ctc_data, color='tab:green', marker='x', label='CTC')
    plt.plot(x, sted_data, color='tab:blue', marker='x', label='STED')
    plt.xticks(x, ticklabels)
    plt.xlabel("Label %")
    plt.ylabel("Accuracy")
    plt.legend()
    fig.savefig(f'/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/evaluation/results/{args.model}/finetuned/{args.dataset}_fewshot.pdf', bbox_inches='tight', dpi=1200)




def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {device} ---")
    print(f"--- Evaluating on {args.dataset} ---")
    accuracies = {
        "ImageNet": [],
        "CTC": [],
        "STED": []
    }
    blocks = 'all' if args.probe_type == 'linear-probe' else "0"
    percentages = ["1", "10", "25", "50"]

    for pretraining in ["STED"]:
        n_channels = 3 if pretraining == "ImageNet" else 1
        for perc in percentages:
            path = f"{blocks}blocks_{perc}%_labels"
            print(f"--- Evaluating {args.model}_pretraining with {perc}% of labels ---")
            model, cfg = get_classifier_v2(
                name=args.model,
                weights=pretraining,
                task='frozen',
                path=path,
                dataset=args.dataset,
                in_channels=n_channels,
                blocks=blocks
            )
            model = model.to(device)
            _, _, test_loader = get_dataset(
                name=args.dataset,
                transform=None,
                path=None,
                n_channels=n_channels,
                training=True,
                batch_size=cfg.batch_size
            )
            acc = evaluate(model=model, loader=test_loader, device=device)
            accuracies[pretraining].append(acc)


if __name__=="__main__":
    main()