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
parser.add_argument("--pretraining", type=str, default="STED")
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
        print(f"********* {args.model} - {args.pretraining} | {args.task} | {args.dataset} **********")
        print("Overall accuracy = {:.3f}".format(accuracies[0]))
        for i in range(1, 4+1):
            acc = accuracies[i]
            print("Class {} accuracy = {:.3f}".format(
                i, acc))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(f"--- Running on {device} ---")
    n_channels = 3 if args.pretraining == "ImageNet" else 1 
    model = get_classifier(
        name=args.model,
        pretraining=args.pretraining,
        task=args.task,
        dataset=args.dataset
    ).to(device)
    _, _, test_loader = get_dataset(
        name=args.dataset,
        transform=None, 
        path=None, 
        n_channels=n_channels,
        training=True
    )
    evaluate(model=model, loader=test_loader, device=device)


if __name__=="__main__":
    main()