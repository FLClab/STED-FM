
import torch
import numpy
import os
import matplotlib 

from typing import Optional, List
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from matplotlib import pyplot
from tqdm import tqdm

from stedfm.DEFAULTS import BASE_PATH
from stedfm.datasets import get_multichannel_dataset
from stedfm import get_pretrained_model_v2
from stedfm.models.classifier import MetaLinearProbe

cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    name="nice-prism",
    colors=["#5F4690","#1D6996","#38A6A5","#0F8554","#73AF48","#EDAD08","#E17C05","#CC503E","#94346E"]
#     colors=["#45d7cd", "#ff5554"]
#     colors=["#ffc949", "#ff5554"]
#     colors = ["#1b5bdb", "#ffc949"]
)
matplotlib.colormaps.register(cmap=cmap)
matplotlib.colormaps.register(cmap=cmap.reversed())

def plot_PCA(args, samples: numpy.ndarray, labels: numpy.ndarray, classes: Optional[List[str]] = None) -> None:
    pca = PCA(n_components=2, random_state=42)
    pca_features = pca.fit_transform(samples)

    fig, ax = pyplot.subplots(figsize=(5, 5))

    uniques = numpy.unique(labels) 
    cmap = pyplot.get_cmap("nice-prism", len(uniques))
    for i, unique in enumerate(uniques):
        mask = labels == unique
        ax.scatter(
            pca_features[mask, 0], pca_features[mask, 1], 
            color=cmap(i), 
            label=labels[i] if classes is None else classes[i], 
            alpha=0.5 if classes is not None and "Control" in classes[i] else 1.0
        )
      
    ax.set(
        ylabel="PCA-2", xlabel="PCA-1",
        xticks=[], yticks=[]
    )
    ax.legend()
    os.makedirs("./results/knn-pca", exist_ok=True)
    fig.savefig(f"./results/knn-pca/pca_{args.dataset}_{args.weights}.pdf", bbox_inches="tight")
    pyplot.close()

def plot_umap(args, samples: numpy.ndarray, labels: numpy.ndarray, classes: Optional[List[str]] = None) -> None:
    import umap
    reducer = umap.UMAP(random_state=42)
    umap_features = reducer.fit_transform(samples)

    fig, ax = pyplot.subplots(figsize=(5, 5))

    uniques = numpy.unique(labels) 
    cmap = pyplot.get_cmap("nice-prism", len(uniques))
    for i, unique in enumerate(uniques):
        mask = labels == unique
        ax.scatter(
            umap_features[mask, 0], umap_features[mask, 1], 
            color=cmap(i), 
            label=labels[i] if classes is None else classes[i], 
            alpha=0.5 if classes is not None and "Control" in classes[i] else 1.0
        )
      
    ax.set(
        ylabel="UMAP-2", xlabel="UMAP-1",
        xticks=[], yticks=[]
    )
    ax.legend()
    os.makedirs("./results/knn-umap", exist_ok=True)
    fig.savefig(f"./results/knn-umap/umap_{args.dataset}_{args.weights}.pdf", bbox_inches="tight")
    pyplot.close()

def extract_features(args, model: torch.nn.Module, loader: DataLoader, device: torch.device):
    samples, ground_truth = [], []
    with torch.no_grad():
        for x, data_dict in tqdm(loader, desc="Extracting features..."):
            labels = data_dict['label']
            x, labels = x.to(device), labels.to(device)
            if "mcms" in args.model.lower():
                features = model.forward_features(x, pixel_size=data_dict.get("pixel-size", None))
            elif "mae" in args.model.lower():
                features = model.forward_features(x)
            elif "convnext" in args.model.lower():
                features = model(x).flatten(start_dim=1)
            else:
                features = model(x)

            truth = labels.data.cpu().numpy()
            feat = features.data.cpu().numpy()
            ground_truth.extend(truth)
            samples.extend(feat)
    samples = numpy.array(samples)
    ground_truth = numpy.array(ground_truth).astype(numpy.int64)
    return samples, ground_truth
                
def knn_predict(args, model: torch.nn.Module, valid_loader: DataLoader, test_loader: DataLoader, device:torch.device, savename:str) -> None:
    valid_samples, valid_ground_truth = extract_features(args=args, model=model, loader=valid_loader, device=device)
    test_samples, test_ground_truth = extract_features(args=args, model=model, loader=test_loader, device=device)

    if args.pca:
        merged_samples = numpy.concatenate([valid_samples, test_samples], axis=0)
        merged_labels = numpy.concatenate([valid_ground_truth, test_ground_truth], axis=0)
        plot_PCA(args=args, samples=merged_samples, labels=merged_labels, classes=test_loader.dataset.classes)

    if args.umap:
        merged_samples = numpy.concatenate([valid_samples, test_samples], axis=0)
        merged_labels = numpy.concatenate([valid_ground_truth, test_ground_truth], axis=0)
        plot_umap(args=args, samples=merged_samples, labels=merged_labels, classes=test_loader.dataset.classes)

    pdistances = cdist(valid_samples, test_samples, metric="cosine").T
    neighbor_indices = numpy.argsort(pdistances, axis=1)
    mean_accuracies = []
    for n in list(range(3, 11)):
        neighbors = neighbor_indices[:, :n]

        associated_labels = valid_ground_truth[neighbors]
        uniques = numpy.unique(valid_ground_truth).astype(numpy.int64)

        confusion_matrix = numpy.zeros((len(uniques), len(uniques)))

        for neighbor_labels, truth in zip(associated_labels, test_ground_truth):
            votes, vote_counts = numpy.unique(neighbor_labels, return_counts=True)
            max_idx = numpy.argmax(vote_counts)
            max_vote = votes[max_idx]
            vote_count = vote_counts[max_idx]
            if vote_count > 1: # Given our 4-class problems, this should always be true, but useful if ever we do more than 4 classes
                confusion_matrix[truth, max_vote] += 1 
        
        # print("--- Confusion matrix ---")
        accuracy = numpy.diag(confusion_matrix).sum() / test_ground_truth.shape[0]
        accuracies = []
        accuracies.append(accuracy)
        # print(f"--- {args.dataset} ; {args.model} ; {savename} ---\n")
        for i in range(len(uniques)):
            N = numpy.sum(confusion_matrix[i, :])
            correct = confusion_matrix[i, i] 
            class_acc = correct / N
            accuracies.append(class_acc)
        mean_accuracies.append(accuracy)
        accuracies = numpy.array(accuracies)
    print(f"\n--- {args.dataset} ; {args.model} ; {savename} ---")
    print(f"\tAverage accuracy: {numpy.mean(mean_accuracies) * 100:0.2f} ± {numpy.std(mean_accuracies) * 100:0.2f}")
    print(f"\tMaximum accuracy: {numpy.max(mean_accuracies) * 100:0.2f}")
    print(f"\tMinimum accuracy: {numpy.min(mean_accuracies) * 100:0.2f}\n")

def main():
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mRNAs-3b")
    parser.add_argument("--model", type=str, default="mae-mcms-lightning-small")
    parser.add_argument("--weights", type=str, default="MAE_MCMS_SMALL_STED")
    parser.add_argument("--global-pool", type=str, default="avg")
    parser.add_argument("--pca", action="store_true")
    parser.add_argument("--umap", action="store_true")
    parser.add_argument("--n-neighbors", type=int, default=11)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, cfg = get_pretrained_model_v2(
        name=args.model, 
        weights=args.weights, 
        as_classifier=True,
        blocks="all",
        classifier_type=MetaLinearProbe,
        channel_token_pool="avg",
    )
    model.to(device)
    model.eval()
    
    train_dataset, valid_dataset, test_dataset = get_multichannel_dataset(args.dataset, cfg)
    print(f"Dataset loaded: {args.dataset}")
    print(f"\tNumber of training samples: {len(train_dataset)}")
    print(f"\tNumber of validation samples: {len(valid_dataset)}")
    print(f"\tNumber of test samples: {len(test_dataset)}")

    # Since images can have different sizes, we use a batch size of 1 and extract features for each image independently before applying KNN on the extracted features.
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    with torch.no_grad():
        knn_predict(args=args, model=model, valid_loader=train_loader, test_loader=test_loader, device=device, savename=f"{args.model}_{args.weights}_neighbors-{args.n_neighbors}")

if __name__ == "__main__":

    main()