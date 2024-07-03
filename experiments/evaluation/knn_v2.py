import torch 
import argparse 
import matplotlib.pyplot as plt
import sys 
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm 
from sklearn.neighbors import NearestNeighbors 
from torchvision import transforms
import numpy as np
import seaborn 
from scipy.spatial.distance import pdist, cdist
import pandas 
sys.path.insert(0, "../")
from loaders import get_dataset 
from model_builder import get_pretrained_model_v2 

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="synaptic-proteins")
parser.add_argument("--model", type=str, default="mae-small")
parser.add_argument("--weights", type=str, default="MAE_TINY_STED")
parser.add_argument("--global-pool", type=str, default="avg")
parser.add_argument("--pca", action="store_true")
args = parser.parse_args()

def plot_PCA(samples: np.ndarray, labels: np.ndarray, savename: str) -> None:
    pass 

def knn_predict(model: torch.nn.Module, loader: DataLoader, device:torch.device, savename:str) -> None:
    samples, ground_truth = [], []
    with torch.no_grad():
        for x, data_dict in tqdm(loader, desc="Extracting features..."):
            labels = data_dict['label']
            x, labels = x.to(device), labels.to(device) 
            if "mae" in args.model.lower():
                features = model.forward_encoder(x)
                if args.global_pool == "token":
                    features = features[:, 0, :] # class token
                else:
                    features = torch.mean(features[:, 1:, :], dim=1) # average patch tokens
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

    if args.pca:
        plot_PCA(samples=samples, labels=ground_truth, savename=savename)

    neighbors_obj = NearestNeighbors(n_neighbors=6, metric="precomputed")
    pdistances = cdist(samples, samples)
    neighbors_obj = neighbors_obj.fit(pdistances)
    distances, neighbors = neighbors_obj.kneighbors(X=pdistances, return_distance=True)
    neighbors = neighbors[:, 1:]

    associated_labels = ground_truth[neighbors]
    uniques = np.unique(ground_truth).astype(np.int64)
    
    confusion_matrix = np.zeros((len(uniques), len(uniques)))
    for unique in tqdm(uniques, desc="Confusion matrix computation..."):
        mask = ground_truth == unique
        for predicted_unique in uniques:
            votes = np.sum((associated_labels[mask] == predicted_unique).astype(int), axis=-1)
            confusion_matrix[unique, predicted_unique] += np.sum(votes >= 3)
    accuracy = np.diag(confusion_matrix).sum() / np.sum(confusion_matrix)
    print(f"--- {args.dataset} ; {args.model} ; {savename} ---\n\tAccuracy: {accuracy * 100:0.2f}\n")
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
    )
    ax.set_title(round(acc, 4))
    fig.savefig(f"./results/{args.model}/{savename}_{args.dataset}_knn_results.pdf", dpi=1200, bbox_inches='tight', transparent=True)
    plt.close(fig)


def get_save_folder() -> str: 
    if "imagenet" in args.weights.lower():
        return "ImageNet"
    elif "sted" in args.weights.lower():
        return "STED"
    elif "jump" in args.weights.lower():
        return "JUMP"
    elif "ctc" in args.weights.lower():
        return "CTC"
    elif "hpa" in args.weights.lower():
        return "HPA"
    else:
        raise NotImplementedError("The requested weights do not exist.")
    
def main():
    SAVE_NAME = get_save_folder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {device} ---")
    n_channels = 3 if SAVE_NAME == "ImageNet" else 1
    model, cfg = get_pretrained_model_v2(
        name=args.model,
        weights=args.weights,
        path=None,
        mask_ratio=0.0,
        pretrained=True if SAVE_NAME == "ImageNet" else False,
        in_channels=n_channels,
        as_classifier=False,
        blocks='0' # Not used with as_classifier = False
    ) 

    _, _, loader = get_dataset(
        name=args.dataset,
        transform=None,
        path=None,
        n_channels=n_channels,
        training=True,
        batch_size=64,
        num_samples=None # Not used when only getting test dataset
    )

    model = model.to(device)
    model.eval()
    knn_predict(model=model, loader=loader, device=device, savename=SAVE_NAME)

if __name__=="__main__":
    main()
