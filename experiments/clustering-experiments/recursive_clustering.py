import numpy as np 
import matplotlib.pyplot as plt 
import torch 
from torch.utils.data import DataLoader 
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
from tqdm import tqdm, trange 
from sklearn.cluster import KMeans 
import sys 
import bisect 
import os 
import pickle
from itertools import combinations
import argparse 
from stedfm.DEFAULTS import BASE_PATH  
from stedfm.loaders import get_dataset 
from skimage import measure, feature
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from stedfm import get_pretrained_model_v2

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, default="neural-activity-states")
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--embed", action="store_true")
parser.add_argument("--mode", type=str, default="deep", choices=["deep", "manual", "morphological"])
args = parser.parse_args()

PIXELSIZE = 0.015 # in um

class ConsensusCluster:
    """
      Implementation of Consensus clustering, following the paper
      https://link.springer.com/content/pdf/10.1023%2FA%3A1023949509487.pdf
      Args:
        * cluster -> clustering class
        * NOTE: the class is to be instantiated with parameter `n_clusters`,
          and possess a `fit_predict` method, which is invoked on data.
        * L -> smallest number of clusters to try
        * K -> biggest number of clusters to try
        * H -> number of resamplings for each cluster number
        * resample_proportion -> percentage to sample
        * Mk -> consensus matrices for each k (shape =(K,data.shape[0],data.shape[0]))
                (NOTE: every consensus matrix is retained, like specified in the paper)
        * Ak -> area under CDF for each number of clusters 
                (see paper: section 3.3.1. Consensus distribution.)
        * deltaK -> changes in areas under CDF
                (see paper: section 3.3.1. Consensus distribution.)
        * self.bestK -> number of clusters that was found to be best
      """

    def __init__(self, cluster, L, K, H, resample_proportion=0.5, random_state=None):
        assert 0 <= resample_proportion <= 1, "proportion has to be between 0 and 1"
        self.cluster_ = cluster
        self.resample_proportion_ = resample_proportion
        self.L_ = L
        self.K_ = K
        self.H_ = H
        self.random_state_ = random_state
        self.Mk = None
        self.Ak = None
        self.deltaK = None
        self.bestK = None

        self.rng = np.random.RandomState(self.random_state_)

    def _internal_resample(self, data, proportion):
        """
        Args:
          * data -> (examples,attributes) format
          * proportion -> percentage to sample
        """
        resampled_indices = np.sort(self.rng.choice(
            range(data.shape[0]), size=int(data.shape[0]*proportion), replace=False))
        return resampled_indices, data[resampled_indices, :]

    def fit(self, data, verbose=False):
        """
        Fits a consensus matrix for each number of clusters

        Args:
          * data -> (examples,attributes) format
          * verbose -> should print or not
        """
        Mk = np.zeros((self.K_-self.L_, data.shape[0], data.shape[0]))
        Is = np.zeros((data.shape[0],)*2)
        for k in range(self.L_, self.K_):  # for each number of clusters
            i_ = k-self.L_
            if verbose:
                print("At k = %d, aka. iteration = %d" % (k, i_))
            for h in range(self.H_):  # resample H times
                if verbose:
                    print("\tAt resampling h = %d, (k = %d)" % (h, k))
                resampled_indices, resample_data = self._internal_resample(
                    data, self.resample_proportion_)
                Mh = self.cluster_(n_clusters=k, n_init=10,random_state=self.random_state_ if self.random_state_ is None else self.random_state_ + h).fit_predict(resample_data)
                # find indexes of elements from same clusters with bisection
                # on sorted array => this is more efficient than brute force search
                index_mapping = np.array((Mh, resampled_indices)).T
                index_mapping = index_mapping[index_mapping[:, 0].argsort()]
                sorted_ = index_mapping[:, 0]
                id_clusts = index_mapping[:, 1]
                for i in range(k):  # for each cluster
                    ia = bisect.bisect_left(sorted_, i)
                    ib = bisect.bisect_right(sorted_, i)
                    is_ = np.sort(id_clusts[ia:ib])
                    ids_ = np.array(list(combinations(is_, 2))).T
                    # sometimes only one element is in a cluster (no combinations)
                    if ids_.size != 0:
                        Mk[i_, ids_[0], ids_[1]] += 1
                # increment counts
                ids_2 = np.array(list(combinations(resampled_indices, 2))).T
                Is[ids_2[0], ids_2[1]] += 1
            Mk[i_] /= Is+1e-8  # consensus matrix
            # Mk[i_] is upper triangular (with zeros on diagonal), we now make it symmetric
            Mk[i_] += Mk[i_].T
            Mk[i_, range(data.shape[0]), range(
                data.shape[0])] = 1  # always with self
            Is.fill(0)  # reset counter
        self.Mk = Mk
        # fits areas under the CDFs
        self.Ak = np.zeros(self.K_-self.L_)
        for i, m in enumerate(Mk):
            hist, bins = np.histogram(m.ravel(), density=True)
            self.Ak[i] = sum(h*(b-a)
                             for b, a, h in zip(bins[1:], bins[:-1], np.cumsum(hist)))
        # fits differences between areas under CDFs
        self.deltaK = np.array([(Ab-Aa)/Aa if i > 2 else Aa
                                for Ab, Aa, i in zip(self.Ak[1:], self.Ak[:-1], range(self.L_, self.K_-1))])
        self.bestK = np.argmax(self.deltaK) + \
            self.L_ if self.deltaK.size > 0 else self.L_

    def predict(self):
        """
        Predicts on the consensus matrix, for best found cluster number
        """
        assert self.Mk is not None, "First run fit"
        return self.cluster_(n_clusters=self.bestK).fit_predict(
            1-self.Mk[self.bestK-self.L_])

    def intra_cluster_stability(self):
        assert self.Mk is not None, "First run fit"
        intra_cluster_stability = np.zeros(self.K_-self.L_)
        for k in range(self.L_, self.K_):
            Mk = self.Mk[k-self.L_]
            labels = self.cluster_(n_clusters=k).fit_predict(1 - Mk)
            for i in range(k):
                mask = labels == i
                intra_cluster_stability[k-self.L_] += Mk[mask][:, mask].mean()
        return intra_cluster_stability / np.arange(self.L_, self.K_)

    def inter_cluster_overlap(self):
        assert self.Mk is not None, "First run fit"
        inter_cluster_overlap = np.zeros(self.K_-self.L_)
        denominator = np.zeros(self.K_-self.L_)
        for k in range(self.L_, self.K_):
            Mk = self.Mk[k-self.L_]
            labels = self.cluster_(n_clusters=k).fit_predict(1 - Mk)
            for i, j in combinations(range(k), 2):
                mask_i = labels == i
                mask_j = labels == j
                inter_cluster_overlap[k-self.L_] += Mk[mask_i][:, mask_j].mean()
                denominator[k-self.L_] += 1
        return inter_cluster_overlap / denominator

    def predict_data(self, data):
        """
        Predicts on the data, for best found cluster number
        Args:
          * data -> (examples,attributes) format 
        """
        assert self.Mk is not None, "First run fit"
        return self.cluster_(n_clusters=self.bestK).fit_predict(data)
    
def unravel_clusters(clusters):
    if isinstance(clusters, list):
        for cluster in clusters:
            yield from unravel_clusters(cluster)
    else:
        yield clusters

def recursive_cluster(items, metadata=None, depth=0):
    print(f"==== NOW AT DEPTH: {depth} ====")
    if len(items) < 10 / 0.8:
        return {"data": items, "metadata": metadata}
    consensus_cluster = ConsensusCluster(KMeans, 2, 10, 30, resample_proportion=0.8, random_state=42)
    consensus_cluster.fit(items, verbose=True)
    best_k = consensus_cluster.bestK
    intra_cluster_stability = consensus_cluster.intra_cluster_stability()
    inter_cluster_overlap = consensus_cluster.inter_cluster_overlap()
    print(f"[{depth:->4}] Number of items: {len(items)}; Best K: {best_k}; Intra Cluster Stability: {intra_cluster_stability[best_k - 2]:0.3f}; Inter Cluster Overlap: {inter_cluster_overlap[best_k - 2]:0.3f}")
    if intra_cluster_stability[best_k - 2] < 0.8 or inter_cluster_overlap[best_k - 2] > 0.2:
        return {"data": items, "metadata": metadata}
    else:
        clusters = []
        labels = consensus_cluster.predict()
        for i in range(best_k):
            mask = labels == i
            masked_metadata = [metadata[j] for j in range(len(metadata)) if mask[j]] if metadata is not None else None
            clusters.append(recursive_cluster(items[mask], masked_metadata, depth=depth+1))
        return clusters

def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def embed_dataset(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> torch.Tensor:
    embeddings, labels = [], [] 
    dataset_indices = []
    conditions = []
    with torch.no_grad():
        for img, metadata in tqdm(dataloader, desc="Embedding dataset.."):
            img = img.to(device)
            label = metadata["label"] 
            data_index = metadata["dataset-idx"]
            if args.dataset == "neural-activity-states":
                condition = metadata["condition"]
                conditions.extend(condition)
            output = model.forward_features(img)
            embeddings.extend(output.data.cpu().numpy())
            labels.extend(label)
            dataset_indices.extend(data_index)
    embeddings = np.array(embeddings)
    return embeddings, labels, dataset_indices, conditions

def extract_features(dataset: torch.utils.data.Dataset):
    features = []
    labels = []
    dataset_indices = []
    conditions = []
    for img, metadata in tqdm(dataset, desc="Extracting features.."):
        img = img.squeeze().numpy()
        mask = metadata["mask"] 
        condition = metadata["condition"]
        dataset_idx = metadata["dataset-idx"]
        label = metadata["label"] # Note that for the neural activity states dataset, label == condition
        img_features = get_handcrafted_features(img, mask)
        if img_features is None:
            continue
        features.append(img_features)
        labels.append(label)
        conditions.append(condition)
        dataset_indices.append(dataset_idx)
    features = np.array(features)
    # labels = np.array(labels)
    # conditions = np.array(conditions)
    # dataset_indices = np.array(dataset_indices)

    return features, labels, dataset_indices, conditions


# def get_average_features(img, mask):
#     area, intensity, eccentricity, nn, num_proteins, density, blur_effect, shannon_entropy, signal_to_noise = [], [], [], [], [], [], [], [], []

#     label_image, nprots = measure.label(mask, return_num=True)
#     # Image-level features
#     num_proteins.append(nprots)
#     br = measure.blur_effect(img)
#     blur_effect.append(br)
#     shannon_entropy.append(measure.shannon_entropy(img))
#     foreground_intensity = np.mean(img[mask])
#     inverted_mask = np.logical_not(mask)
#     background_intensity = np.mean(img[inverted_mask])
#     signal_to_noise.append(foreground_intensity / background_intensity)
 
#     rprops = measure.regionprops(label_image, intensity_image=img)
#     centroids = [r.weighted_centroid for r in rprops]
#     if len(centroids) == 1:
#         density.append(1.0)
#     else:
#         distance_matrix = distance.cdist(centroids, centroids, metric="euclidean")
#         distance_matrix = np.sort(distance_matrix, axis=1)
#         img_density = []
#         for d in range(distance_matrix.shape[0]):
#             num_neighbors = np.sum(distance_matrix[d] < 50)
#             img_density.append(num_neighbors)
#         density.append(np.mean(img_density))
        
#     img_area, img_intensity, img_eccentricity, img_nn = [], [], [], []
#     for rprop in rprops:
#         distances = distance.cdist(centroids, [rprop.weighted_centroid], metric="euclidean")
#         distances = distances[1:]
#         if distances.shape[0] > 0:
#             img_nn.append(np.min(distances))
#         else:
#             img_nn.append(224.0)
#         img_area.append(rprop.area)
#         img_intensity.append(rprop.mean_intensity)
#         img_eccentricity.append(rprop.eccentricity)
#     area.append(np.mean(img_area))
#     intensity.append(np.mean(img_intensity))
#     eccentricity.append(np.mean(img_eccentricity))
#     nn.append(np.mean(img_nn))
#     nn = [value for value in nn if not np.isnan(value)]
#     area = np.mean(area)
#     intensity = np.mean(intensity)
#     eccentricity = np.mean(eccentricity)
#     nn = np.mean(nn)
#     num_proteins = np.mean(num_proteins)
#     density = np.mean(density)
#     blur_effect = np.mean(blur_effect)
#     shannon_entropy = np.mean(shannon_entropy)
#     signal_to_noise = np.mean(signal_to_noise)
#     average_features = np.array([area, intensity, eccentricity, nn, num_proteins, density, blur_effect, shannon_entropy, signal_to_noise])
#     if np.any(np.isnan(average_features)):
#         # print("\n\n")
#         # print("NaN features")
#         # print(average_features)
#         # print("\n\n")
#         return None
#     return average_features

def get_handcrafted_features(img, mask):
    mask_label, num_proteins = measure.label(mask, return_num=True)
    props = measure.regionprops(mask_label, intensity_image=img)
    coordinates = np.array([p.weighted_centroid for p in props])
    distances = distance.pdist(coordinates) * PIXELSIZE 
    distances = distance.squareform(distances)

    if distances.shape[0] <= 1:
        nn_distances = [30.0]
    else:
        nn_distances = np.sort(distances, axis=1)[:, 1]
    density_proteins = (np.sum(distances < 0.5, axis=1) - 1) / 1 
    br = measure.blur_effect(img) # 1 when blurry, 0 when sharp 
    shannon_entropy = measure.shannon_entropy((img))
    foreground_intensity = np.mean(img[mask])
    inverted_mask = np.logical_not(mask) 
    background_intensity = np.mean(img[inverted_mask])
    snr = (foreground_intensity - background_intensity) / background_intensity 
    features = []
    for prop, density, nn in zip(props, density_proteins, nn_distances):
        if np.isnan(nn):
            nn = 30.0
        label = prop.image 
        min_distance = int(0.08 / PIXELSIZE) // 2 + 1 
        peaks = feature.peak_local_max(img, min_distance=min_distance, exclude_border=False, labels=label)
        features.append([
            prop.area,
            prop.perimeter,
            prop.eccentricity,
            prop.major_axis_length,
            prop.minor_axis_length,
        ]) 
    features = np.array(features)
    features = np.mean(features, axis=0)
    return features



def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(args.seed)
    print(f"--- Running on {DEVICE} ---")
    n_channels = 3 if "imagenet" in args.weights.lower() else 1
    model, cfg = get_pretrained_model_v2(
        name=args.model,
        weights=args.weights,
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

    train_loader, valid_loader, test_loader = get_dataset(
        name=args.dataset,
        transform=None,
        training=True,
        path=None, 
        batch_size=cfg.batch_size,
        n_channels=n_channels,
        balance=False,
    )

    if args.embed: 
        if args.mode == "deep":
            embeddings, labels, dataset_indices, conditions = embed_dataset(
                model=model,
                dataloader=test_loader,
                device=DEVICE
            )
        else:
            embeddings, labels, dataset_indices, conditions = extract_features(
                dataset=test_loader.dataset,
            )
            embeddings = StandardScaler().fit_transform(embeddings)
        print(embeddings.shape, len(labels), len(dataset_indices), len(conditions))
    else:
        raise NotImplementedError("Pre-computed embeddings not implemented")
    
    if args.dataset == "neural-activity-states":
        metadata = [{"condition": c, "label": l, "dataset-idx": i} for c, l, i in zip(conditions, labels, dataset_indices)]
    else:
        metadata = [{"label": l, "dataset-idx": i} for l, i in zip(labels, dataset_indices)]


    clusters = recursive_cluster(embeddings, metadata=metadata)
    os.makedirs(f"./recursive-clustering-experiment/{args.mode}", exist_ok=True)
    with open(f"./recursive-clustering-experiment/{args.mode}/{args.weights}_{args.dataset}_recursive_clusters_tree.pkl", "wb") as handle:
        pickle.dump(clusters, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
