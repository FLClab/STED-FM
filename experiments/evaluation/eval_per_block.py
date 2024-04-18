import torch 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse 
import sys 
sys.path.insert(0, "../")
from model_builder import get_classifier 
from loaders import get_dataset
from utils import compute_Nary_accuracy

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="MAE")
parser.add_argument("--num-blocks", type=int, default=12)
parser.add_argument("--dataset", type=str, default='optim')
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
    # reverse because the order is in terms of # of blocks frozen, we want # of blocks fine-tuned
    imnet_data = list(reversed(accuracies['ImageNet']))
    ctc_data = list(reversed(accuracies['CTC']))
    sted_data = list(reversed(accuracies["STED"]))
    x = np.arange(0, len(sted_data), 1)
    ticklabels = [str(item + 1) for item in x]
    fig = plt.figure()
    plt.plot(x, imnet_data, color='tab:red', marker='x', label="ImageNet")
    plt.plot(x, ctc_data, color='tab:green', marker='x', label='CTC')
    plt.plot(x, sted_data, color='tab:blue', marker='x', label='STED')
    plt.xticks(x, ticklabels)
    plt.xlabel("# blocks fine-tuned")
    plt.ylabel("Accuracy")
    plt.legend()
    fig.savefig(f'/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/evaluation/results/MAEClassifier/finetuned/{args.dataset}_acc_vs_blocks.pdf', bbox_inches='tight', dpi=1200)


def main():
    blocks_list = list(range(1, args.num_blocks + 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {device} ---")
    accuracies = {
        "ImageNet": [],
        "CTC": [],
        "STED": []
    }

    for pretraining in ["ImageNet", "CTC", "STED"]:
        n_channels = 3 if pretraining == "ImageNet" else 1 
        for bnum in blocks_list:
            print(f"--- Evaluating {args.model}_{pretraining} with {bnum} frozen blocks ---")
            model = get_classifier(
                name=args.model,
                pretraining=pretraining,
                task='finetuned',
                path=f"{bnum}blocks",
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