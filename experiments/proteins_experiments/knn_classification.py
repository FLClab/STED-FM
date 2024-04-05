import numpy as np
from tqdm import tqdm 
import torch
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict, OrderedDict
from utils.data_utils import load_theresa_proteins
from models.intensity_model import get_intensity_model
import argparse
plt.style.use("dark_background")

parser = argparse.ArgumentParser()
parser.add_argument("--class-type", "-ct", type=str, default="protein")
parser.add_argument("--datapath", type=str)
parser.add_argument("--pretraining", type=str, default="STED")
args = parser.parse_args()


def load_model(pretraining: str = "STED"):
    if pretraining == "STED":
        backbone = torchvision.models.resnet18()
        backbone.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=1, bias=False)
        backbone.fc = torch.nn.Identity()
        checkpoint = torch.load("/home/frederic/Datasets/FLCDataset/baselines/result.pt")
        print(checkpoint.keys())
        model_dict = checkpoint["model"]["backbone"]
        backbone.load_state_dict(model_dict)
    elif pretraining == "ImageNet":
        print("Loading ResNet-18 pre-trained on ImageNet")
        backbone = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        backbone.fc = torch.nn.Identity()
    elif pretraining == "Intensity":
        print("Loading IntensityModel")
        backbone = get_intensity_model()
    else:
        exit(f"Model {args.pretraining} not implemented yet.")
    return backbone

def knn_predict(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    out = defaultdict(list)
    with torch.no_grad():
        for x, proteins, conditions in tqdm(loader, desc="Extracting features..."):
            labels = proteins if args.class_type == "protein" else conditions
            x, labels = x.to(device), labels.to(device)
            features = model(x)
            out["features"].extend(features.cpu().data.numpy())
            out["labels"].extend(labels.cpu().data.numpy().tolist())
    samples = np.array(out['features'])
    labels = np.array([int(item) for item in out['labels']])
    neigh = NearestNeighbors(n_neighbors=6)
    neigh.fit(samples)
    neighbors = neigh.kneighbors(samples, return_distance=False)[:, 1:]
    associated_labels = labels[neighbors]

    uniques = np.unique(labels)
    print(np.unique(labels, return_counts=True))
    confusion_matrix = np.zeros((len(uniques), len(uniques)))
    for unique in tqdm(uniques, desc="Confusion matrix computation..."):
        mask = labels == unique
        for predicted_unique in uniques:
            votes = np.sum((associated_labels[mask] == predicted_unique).astype(int), axis=-1)
            confusion_matrix[unique, predicted_unique] += np.sum(votes >= 3)
    accuracy = np.diag(confusion_matrix).sum() / np.sum(confusion_matrix)
    print(f"Accuracy: {accuracy * 100:0.2f}")
    acc = accuracy * 100
    fig, ax = plt.subplots()
    cm = confusion_matrix / np.sum(confusion_matrix, axis=-1)[np.newaxis]
    ax.imshow(cm, vmin=0, vmax=1, cmap="Purples")
    for j in range(cm.shape[-2]):
        for i in range(cm.shape[-1]):
            ax.annotate(
                f"{cm[j, i]:0.2f}\n({confusion_matrix[j, i]:0.0f})", (i, j), 
                horizontalalignment="center", verticalalignment="center",
                color="white" if cm[j, i] > 0.5 else "black"
            )
    ax.set(
        xticks=uniques, yticks=uniques,
        # xticklabels=['Round', 'Elongated', 'Multidomain'],
        # yticklabels=['Round', 'Elongated', 'Multidomain']    
    )
    ax.set_title(round(acc, 4))
    fig.savefig(f"./results/{args.pretraining}/{args.class_type}_knn_results.pdf", dpi=1200, bbox_inches='tight', transparent=True)
    plt.close(fig)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(pretraining=args.pretraining).to(device)
    model.eval()
    loader = load_theresa_proteins(
        path="/home/frederic/Datasets/FLCDataset",
        class_type=args.class_type,
        n_channels=1 if args.pretraining == "STED" else 3,
    )
    knn_predict(model=model, loader=loader, device=device)

if __name__=="__main__":
    main()