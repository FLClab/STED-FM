import torch 
import os
import argparse 
import matplotlib.pyplot as plt
import sys 
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm 
from torchvision import transforms
import numpy as np
from sklearn.decomposition import PCA
import seaborn 
import json
from scipy.spatial.distance import pdist, cdist
from typing import List, Optional
import pandas 
from stedfm.DEFAULTS import BASE_PATH
from stedfm.loaders import get_dataset 
from stedfm import get_pretrained_model_v2 

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="optim")
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--global-pool", type=str, default="avg")
parser.add_argument("--pca", action="store_true")
parser.add_argument("--n-neighbors", type=int, default=10)
args = parser.parse_args()

def plot_PCA(samples: np.ndarray, labels: np.ndarray, classes: Optional[List[str]] = None) -> None:
    pca = PCA(n_components=2, random_state=42)
    pca_features = pca.fit_transform(samples)

    fig, ax = plt.subplots(figsize=(4, 4))

    uniques = np.unique(labels) 
    cmap = plt.get_cmap("tab10", len(uniques))
    for i, unique in enumerate(uniques):
        mask = labels == unique
        ax.scatter(pca_features[mask, 0], pca_features[mask, 1], color=cmap(i), label=labels[i] if classes is None else classes[i])
      
    ax.set(
        ylabel="PCA-2", xlabel="PCA-1",
        xticks=[], yticks=[]
    )
    ax.legend()
    fig.savefig(f"./results/knn-pca/pca_{args.dataset}_{args.weights}.pdf", bbox_inches="tight")
    plt.close()

def extract_features(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    samples, ground_truth = [], []
    with torch.no_grad():
        for x, data_dict in tqdm(loader, desc="Extracting features..."):
            labels = data_dict['label']
            x, labels = x.to(device), labels.to(device)
            if "mae" in args.model.lower():
                features = model.forward_features(x)
                # if args.global_pool == "token":
                #     features = features[:, 0, :] # class token 
                # else:
                #     features = torch.mean(features[:, 1:, :], dim=1) # average patch tokens 
            elif "convnext" in args.model.lower():
                features = model(x).flatten(start_dim=1)
            else:
                features = model(x)
            
            truth = labels.data.cpu().numpy()
            feat = features.data.cpu().numpy()
            ground_truth.extend(truth)
            samples.extend(feat)
    samples = np.array(samples)
    ground_truth = np.array(ground_truth).astype(np.int64)
    return samples, ground_truth
                


def knn_predict(model: torch.nn.Module, valid_loader: DataLoader, test_loader: DataLoader, device:torch.device, savename:str) -> None:

    valid_samples, valid_ground_truth = extract_features(model=model, loader=valid_loader, device=device)
    test_samples, test_ground_truth = extract_features(model=model, loader=test_loader, device=device)

    if args.pca:
        plot_PCA(samples=test_samples, labels=test_ground_truth, classes=test_loader.dataset.classes)

    pdistances = cdist(valid_samples, test_samples, metric="cosine").T
    neighbor_indices = np.argsort(pdistances, axis=1)
    mean_accuracies = []
    for n in list(range(3, 11)):
        neighbors = neighbor_indices[:, :n]

        associated_labels = valid_ground_truth[neighbors]
        uniques = np.unique(valid_ground_truth).astype(np.int64)

        confusion_matrix = np.zeros((len(uniques), len(uniques)))

        for neighbor_labels, truth in zip(associated_labels, test_ground_truth):
            votes, vote_counts = np.unique(neighbor_labels, return_counts=True)
            max_idx = np.argmax(vote_counts)
            max_vote = votes[max_idx]
            vote_count = vote_counts[max_idx]
            if vote_count > 1: # Given our 4-class problems, this should always be true, but useful if ever we do more than 4 classes
                confusion_matrix[truth, max_vote] += 1 
                

        # print("--- Confusion matrix ---")
        # print(confusion_matrix)
        # print("\n")
        accuracy = np.diag(confusion_matrix).sum() / test_ground_truth.shape[0]

        accuracies = []
        accuracies.append(accuracy)
        # print(f"--- {args.dataset} ; {args.model} ; {savename} ---\n")
        for i in range(len(uniques)):
            N = np.sum(confusion_matrix[i, :])
            correct = confusion_matrix[i, i] 
            class_acc = correct / N
            accuracies.append(class_acc)
            # print(f"\tClass {i} accuracy: {class_acc * 100:0.2f}")

        # print(f"\tOverall accuracy: {accuracy * 100:0.2f}\n")
        mean_accuracies.append(accuracy)
        accuracies = np.array(accuracies)
    print(f"\n--- {args.dataset} ; {args.model} ; {savename} ---")
    print(f"\tAverage accuracy: {np.mean(mean_accuracies) * 100:0.2f} ± {np.std(mean_accuracies) * 100:0.2f}")
    print(f"\tMaximum accuracy: {np.max(mean_accuracies) * 100:0.2f}")
    print(f"\tMinimum accuracy: {np.min(mean_accuracies) * 100:0.2f}\n")

        # np.savez(f"./results/{args.model}/{args.dataset}_{args.weights}_KNN_accuracies.npz", accuracies=accuracies)

        # acc = accuracy * 100
        # fig, ax = plt.subplots()
        # cm = confusion_matrix / np.sum(confusion_matrix, axis=-1)[np.newaxis]
        # ax.imshow(cm, vmin=0, vmax=1, cmap="Purples")
        # for j in range(cm.shape[-2]):
        #     for i in range(cm.shape[-1]):
        #         ax.annotate(
        #             f"{cm[j, i]:0.2f}\n({confusion_matrix[j, i]:0.0f})", (i, j), 
        #             horizontalalignment="center", verticalalignment="center",
        #             color="white" if cm[j, i] > 0.5 else "black"
        #         )
        # ax.set(
        #     xticks=uniques, yticks=uniques,  
        # )
        # ax.set_title(round(acc, 4))
        # os.makedirs(f"./results/{args.model}", exist_ok=True)
        # fig.savefig(f"./results/{args.model}/{savename}_{args.dataset}_knn_results.pdf", dpi=1200, bbox_inches='tight', transparent=True)
        # plt.close(fig)
        # return confusion_matrix


def get_save_folder() -> str: 
    if "imagenet" in args.weights.lower():
        return "ImageNet"
    elif "sted" in args.weights.lower():
        return "STED"
    elif "jump" in args.weights.lower():
        return "JUMP"
    elif "sim" in args.weights.lower():
        return "SIM"
    elif "hpa" in args.weights.lower():
        return "HPA"
    else:
        raise NotImplementedError("The requested weights do not exist.")
    
def main():
    SAVE_NAME = get_save_folder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {device} ---")
    n_channels = 3 if SAVE_NAME == "ImageNet" else 1

    train_loader, valid_loader, test_loader = get_dataset(
        name=args.dataset,
        transform=None,
        training=True,
        path=None,
        n_channels=n_channels,
        batch_size=64,
        num_samples=None, # Not used when only getting test dataset
    )
    

    model, cfg = get_pretrained_model_v2(
        name=args.model,
        weights=args.weights,
        path=None,
        mask_ratio=0.0,
        pretrained=True if SAVE_NAME == "ImageNet" else False,
        in_channels=n_channels,
        as_classifier=True,
        blocks='all',
        num_classes=4, # Ignored, we are not using the classification head
    ) 

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        confusion_matrix = knn_predict(model=model, valid_loader=valid_loader, test_loader=test_loader, device=device, savename=SAVE_NAME)
    
    

if __name__=="__main__":
    main()
