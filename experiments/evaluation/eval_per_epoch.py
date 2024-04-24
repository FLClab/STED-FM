import torch
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import argparse
import sys 
sys.path.insert(0, "../")
from loaders import get_dataset 
from model_builder import get_classifier 
from utils import compute_Nary_accuracy

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="MAE")
parser.add_argument("--task", type=str, default='linear-probe')
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
        # print(f"********* {args.model} - {args.pretraining} | {args.task} | {args.dataset} **********")
        print("Overall accuracy = {:.3f}".format(accuracies[0]))
        for i in range(1, 4+1):
            acc = accuracies[i]
            print("Class {} accuracy = {:.3f}".format(
                i, acc))
    return accuracies[0]

def make_plot(accuracies: dict):
    imnet_data = accuracies["ImageNet"]
    ctc_data = accuracies["CTC"]
    sted_data = accuracies["STED"]
    x = np.arange(0, len(imnet_data), 1)
    ticklabels = [str(item) for item in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
    fig = plt.figure()
    plt.plot(x, imnet_data, color='tab:red', marker='x', label='ImageNet')
    plt.plot(x, ctc_data, color='tab:green', marker='x', label='CTC')
    plt.plot(x, sted_data, color='tab:blue', marker='x', label='STED')
    plt.xticks(x, ticklabels)
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.legend()
    fig.savefig(f'/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/evaluation/results/MAEClassifier/{args.task}/{args.dataset}_acc_vs_epoch.pdf', bbox_inches='tight', dpi=1200)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(f"--- Running on {device} ---")

    accuracies = {
        "ImageNet": [],
        "CTC": [],
        "STED": []
    }

    for pretraining in ["ImageNet", "CTC", "STED"]:
        n_channels = 3 if pretraining == "ImageNet" else 1 
        for path in [f"epoch{item}" for item in [str(2), str(10), str(20), str(30), str(40), str(50), str(60), str(70), str(80), str(90), str(100)]]:
            print(f"--- Evaluating model at {path} ---")
            model = get_classifier(
                name=args.model,
                pretraining=pretraining,
                task=args.task,
                path=path,
                dataset=args.dataset
            ).to(device)
            _, _, test_loader = get_dataset(
                name=args.dataset,
                transform=None, 
                path=None, 
                n_channels=n_channels,
                training=True
            )
            acc = evaluate(model=model, loader=test_loader, device=device)
            accuracies[pretraining].append(acc)

    make_plot(accuracies=accuracies)


if __name__=="__main__":
    main()
