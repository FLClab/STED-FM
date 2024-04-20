import torch
import argparse 
import matplotlib.pyplot as plt 
import sys 
from collections import defaultdict 
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.decomposition import PCA
import seaborn
import pandas
import sys 
sys.path.insert(0, "../")
from loaders import get_dataset
from model_builder import get_pretrained_model, get_pretrained_model_v2

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='synaptic-proteins')
parser.add_argument("--model", type=str, default='MAE')
parser.add_argument("--weights", type=str, default="STED")
parser.add_argument("--global-pool", type=str, default='avg')
args = parser.parse_args()

def plot_PCA(samples, labels, savename):
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
    df = pandas.DataFrame(columns=['PCA-1', 'PCA-2', 'Label'])
    reducer = PCA(n_components=2)
    data = reducer.fit_transform(samples)
    df['PCA-1'] = data[:, 0]
    df['PCA-2'] = data[:, 1]
    df['Label'] = labels
    fig = plt.figure()
    seaborn.scatterplot(data=df, x='PCA-1', y='PCA-2', hue='Label', palette=seaborn.color_palette(colors, 4))
    fig.savefig(f"./results/{args.model}/{savename}_{args.dataset}_PCA.pdf", dpi=1200, bbox_inches='tight', transparent=True)
    plt.close(fig)

def knn_predict(model: torch.nn.Module, loader: DataLoader, device: torch.device, savename: str):
    out = defaultdict(list)
    with torch.no_grad():
        for x, data_dict in tqdm(loader, desc="Extracting features..."):
            labels = data_dict['label']
            x, labels = x.to(device), labels.to(device)
            if args.dataset == "optim":
                labels = labels.type(torch.FloatTensor)

            if args.model in ["mae", "MAE", "MAEClassifier"]:
                features = model.forward_encoder(x)
                if args.global_pool == "token":
                    features = features[:, 0, :] # class token
                else:
                    features = torch.mean(features[:, 1:, :], dim=1) # average patch tokens
            else:
                features = model(x)

    
            out['features'].extend(features.cpu().data.numpy())
            out['labels'].extend(labels.cpu().data.numpy())

    samples = np.array(out['features'])
    labels = np.array([int(item) for item in out['labels']])
    plot_PCA(samples=samples, labels=labels, savename=savename)
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
    else:
        return "CTC"

def main():
    SAVE_NAME = get_save_folder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {device} ---")
    n_channels = 3 if SAVE_NAME == "ImageNet" else 1
    loader = get_dataset(
        name=args.dataset, 
        transform=None, 
        path=None, 
        n_channels=n_channels,
        training=False
        )
    # model = get_pretrained_model(name=args.model, weights=args.weights, path=None).to(device)
    model, _ = get_pretrained_model_v2(
        name=args.model,
        weights=args.weights,
        mask_ratio=0.0,
        pretrained=True if SAVE_NAME == "ImageNet" else False,
        in_channels=n_channels,
        as_classifier=False, # KNN directly in the model's latent space
        blocks='0'
    )
    model = model.to(device)
    model.eval()
    knn_predict(model=model, loader=loader, device=device, savename=SAVE_NAME)
    

if __name__=="__main__":
    main()