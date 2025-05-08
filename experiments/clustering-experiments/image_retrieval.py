from re import L
import torch 
import numpy as np 
import argparse 
import torch.nn.functional as F 
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from stedfm.DEFAULTS import BASE_PATH 
from stedfm.loaders import get_dataset 
from stedfm import get_pretrained_model_v2 

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--global_pool", type=str, default="avg")
parser.add_argument("--num-neighbors", type=int, default=100)
parser.add_argument("--metric", type=str, default="auc", choices=["auc", "aupr"])
args = parser.parse_args()

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {DEVICE} ---")
    pretraining_datasets = ["MAE_SMALL_STED", "MAE_SMALL_SIM", "MAE_SMALL_HPA", "MAE_SMALL_JUMP", "MAE_SMALL_IMAGENET1K_V1"]
    downstream_datasets = ["optim", "neural-activity-states", "peroxisome", "polymer-rings", "dl-sim"]
    P, D = len(pretraining_datasets), len(downstream_datasets)
    performance_heatmap = np.zeros((P, D))
    for i, weights in enumerate(pretraining_datasets):
        for j, dataset in enumerate(downstream_datasets):
            print(f"--- {weights} ---")
            n_channels = 3 if "imagenet" in weights.lower() else 1
            model, cfg = get_pretrained_model_v2(
                name=args.model, 
                weights=weights,
                path=None,
                mask_ratio=0.0,
                pretrained=True if n_channels == 3 else False,
                in_channels=n_channels,
                as_classifier=True,
                blocks="all",
                num_classes=4
            )
            model.to(DEVICE)
            model.eval() 

            _, _, test_loader = get_dataset(
                name=dataset,
                transform=None,
                training=True,
                path=None,
                batch_size=cfg.batch_size,
                n_channels=n_channels
            )

            # Embed dataset
            embeddings, labels, dataset_indices = [], [], []
            N = len(test_loader.dataset)
            with torch.no_grad():
                for n in range(N):
                    img = test_loader.dataset[n][0].unsqueeze(0).to(DEVICE) 
                    metadata = test_loader.dataset[n][1] 
                    label = metadata["label"]
                    dataset_idx = metadata["dataset-idx"]
                    output = model.forward_features(img)
                    embeddings.append(output)
                    labels.append(label)
                    dataset_indices.append(dataset_idx)
            embeddings = torch.cat(embeddings, dim=0)
            labels = np.array(labels)
            dataset_indices = np.array(dataset_indices)
            assert embeddings.shape[0] == labels.shape[0] == dataset_indices.shape[0]

            # _, ks = np.unique(labels, return_counts=True)
            # K = np.min(ks)
            K = args.num_neighbors
            average_precision = []
            for e in trange(embeddings.shape[0]):
                curr_embedding = embeddings[e]
                curr_label = labels[e]
                curr_dataset_idx = dataset_indices[e]
                similarities = F.cosine_similarity(embeddings, curr_embedding.unsqueeze(0), dim=1).cpu().numpy() 
                sorted_indices = np.argsort(similarities)[::-1] 

                query_labels = []

                ## AUC
                for w in sorted_indices:
                    data_index = dataset_indices[w]
                    query_labels.append(1 if labels[w] == curr_label else 0)
                if np.unique(query_labels).shape[0] == 1 and query_labels[0] == 1:
                    auc = 1.0
                elif np.unique(query_labels).shape[0] == 1 and query_labels[0] == 0:
                    auc = 0.0
                else:
                    if args.metric == "auc":
                        auc = roc_auc_score(query_labels, similarities[sorted_indices])
                    elif args.metric == "aupr":
                        auc = average_precision_score(query_labels, similarities[sorted_indices])
                average_precision.append(auc)

                ## Mean average precision @ K = 20

                # for w in sorted_indices[1:K+1]:
                #     data_index = dataset_indices[w]
                #     query_labels.append(labels[w])
                # apk = 0
                # relevance = query_labels == curr_label 
                # num_relevant = np.sum(relevance)
                # if num_relevant == 0:
                #     average_precision.append(0)
                # else:
                #     for temp_idx in range(1, K+1):
                #         pk = np.sum(relevance[:temp_idx]) / temp_idx 
                #         pk = pk * relevance[temp_idx-1]
                #         apk += pk 
                #     apk /= num_relevant 
                #     average_precision.append(apk)


            performance_heatmap[i, j] = np.mean(average_precision)
    
    normalized_heatmap = performance_heatmap.copy()
    for col in range(D):
        diff = 1.0 - np.max(performance_heatmap[:, col])
        normalized_heatmap[:, col] += diff
    

    np.savez(f"./results/{args.metric}_image_retrieval_results.npz", performance_heatmap=performance_heatmap, normalized_heatmap=normalized_heatmap)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(normalized_heatmap, cmap="RdPu")

    for i in range(P):
        for j in range(D):
            text = f'{performance_heatmap[i, j]:.3f}'
            color = "black" if normalized_heatmap[i, j] < 0.89 else "white"
            ax.text(j, i, text, ha="center", va="center", color=color)

    ax.set_xticks(np.arange(D))
    ax.set_yticks(np.arange(P))
    ax.set_xticklabels(downstream_datasets)
    ax.set_yticklabels(pretraining_datasets)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.colorbar(im)
    fig.savefig(f"./results/{args.metric}_image_retrieval_results.pdf", bbox_inches='tight', dpi=1200)
    plt.close(fig)

if __name__=="__main__":
    main()
