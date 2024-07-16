import torch 
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import argparse 
import sys 
sys.path.insert(0, "../")
from model_builder import get_classifier_v3
from loaders import get_dataset 
from utils import compute_Nary_accuracy 

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--pretraining", type=str, default="STED")
parser.add_argument("--probe", type=str, default="linear-probe", choices=["linear-probe", "finetuned"])
parser.add_argument("--dataset", type=str, default="synaptic-proteins")
args = parser.parse_args()

def evaluate(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        device: torch.device
) -> float:
    model.eval()
    big_correct = np.array([0] * (4+1))
    big_n = np.array([0] * (4+1))
    with torch.no_grad():
        for imgs, data_dict in tqdm(loader, desc="Evaluation..."):
            labels = data_dict['label']
            imgs, labels = imgs.to(device), labels.type(torch.LongTensor).to(device)
            if "supervised" in args.model.lower():
                predictions = model(imgs)
            else:
                predictions, _ = model(imgs)
            correct, n = compute_Nary_accuracy(predictions, labels)
            big_correct = big_correct + correct
            big_n = big_n + n
    accuracies = big_correct / big_n
    print(f"********* {args.model} - {args.pretraining} | {args.probe} | {args.dataset} **********")
    print("Overall accuracy = {:.3f}".format(accuracies[0]))
    for i in range(1, 4+1):
        acc = accuracies[i]
        print("Class {} accuracy = {:.3f}".format(
            i, acc))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(f"--- Running on {device} ---")
    n_channels = 3 if args.pretraining == "ImageNet" else 1 
    model, cfg = get_classifier_v3(
        name=args.model,
        in_channels=n_channels,
        mask_ratio=0.0,
        dataset=args.dataset,
        pretraining=args.pretraining,
        probe=args.probe,
    )
    model = model.to(device)
    _, _, test_loader = get_dataset(
        name=args.dataset,
        transform=None,
        path=None,
        n_channels=n_channels,
        batch_size=cfg.batch_size,
        training=True,
        num_samples=None # not used when only loading test dataset
    )

    evaluate(model=model, loader=test_loader, device=device)


    

if __name__=="__main__":
    main()