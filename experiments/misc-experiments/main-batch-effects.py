
import matplotlib.lines
import numpy
import pandas
import anndata
import warnings
import random
import torch
import argparse
import sys
import os
import json
import matplotlib

from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from phate import PHATE
from umap import UMAP
from matplotlib import pyplot
from scipy.spatial.distance import cdist
from skimage.util import montage
from skimage.io import imsave
from torchvision.transforms.functional import gaussian_blur

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')

    import scanpy
    from scib import metrics

from tiffwrapper import make_composite

sys.path.insert(0, "../")
from DEFAULTS import BASE_PATH
from loaders import get_dataset
from model_builder import get_pretrained_model_v2 
from utils import update_cfg, savefig

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--model", type=str, default='mae-lightning-small')
parser.add_argument("--weights", type=str, default=None)
parser.add_argument("--opts", nargs="+", default=[], 
                    help="Additional configuration options")    
args = parser.parse_args()

# Assert args.opts is a multiple of 2
if len(args.opts) == 1:
    args.opts = args.opts[0].split(" ")
assert len(args.opts) % 2 == 0, "opts must be a multiple of 2"

cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    name="nice-prism",
    colors=["#5F4690","#1D6996","#38A6A5","#0F8554","#73AF48","#EDAD08","#E17C05","#CC503E","#94346E"]
)
matplotlib.colormaps.register(cmap=cmap, force=True)
matplotlib.colormaps.register(cmap=cmap.reversed(), force=True)

def set_seeds():
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)

class PoissonBatchEffect:
    def __init__(self, _lambda):
        self._lambda = _lambda

    @property
    def name(self):
        return f"PoissonBatchEffect({self._lambda})"

    def apply(self, X):
        noise = torch.poisson(torch.ones_like(X) * self._lambda) / 255.0
        return X + noise
    
class GaussianBatchEffect:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    @property
    def name(self):
        return f"GaussianBatchEffect({self.mean}, {self.std})"

    def apply(self, X):
        noise = torch.normal(mean=self.mean, std=self.std, size=X.size())
        if X.size(0) == 1:
            return torch.clamp(X + noise, 0, None)
        return X + noise
    
class ContrastBatchEffect:
    def __init__(self, cutoff, gain=10):
        self.cutoff = cutoff
        self.gain = gain

    @property
    def name(self):
        return f"ContrastBatchEffect({self.cutoff}, {self.gain})"

    def apply(self, X):
        vmin, vmax = X.min(), X.max()
        X = (X - vmin) / (vmax - vmin)
        output = 1 / (1 + torch.exp(self.gain * (self.cutoff - X)))
        return torch.clamp(output * (vmax - vmin) + vmin, vmin, vmax)

class ScaleBatchEffect:
    def __init__(self, scale):
        self.scale = scale

    @property
    def name(self):
        return f"ScaleBatchEffect({self.scale})"

    def apply(self, X):
        return X * self.scale
    
class OffsetBatchEffect:
    def __init__(self, offset):
        self.offset = offset

    @property
    def name(self):
        return f"OffsetBatchEffect({self.offset})"

    def apply(self, X):
        return X + self.offset
    
class GaussianBlurBatchEffect:
    def __init__(self, sigma):
        self.sigma = sigma

    @property
    def name(self):
        return f"GaussianBlurBatchEffect({self.sigma})"

    def apply(self, X):
        return gaussian_blur(X, kernel_size=5, sigma=self.sigma)
    
class IdendityBatchEffect:
    @property
    def name(self):
        return "IdendityBatchEffect()"

    def apply(self, X):
        return X
    
def nmi(adata, label_key, cluster_key):
    """
    Compute the normalized mutual information between the labels and the clusters.

    :param adata: Anndata object
    :param label_key: Key for the labels
    :param cluster_key: Key for the clusters
    :return: NMI value
    """
    return metrics.nmi(adata, label_key, cluster_key)

def ari(adata, label_key, cluster_key):
    """
    Compute the adjusted rand index between the labels and the clusters.

    :param adata: Anndata object
    :param label_key: Key for the labels
    :param cluster_key: Key for the clusters
    :return: ARI value
    """
    return metrics.ari(adata, label_key, cluster_key)

def graph_connectivity(adata, label_key):
    """
    Compute the graph connectivity between the labels and the clusters.

    :param adata: Anndata object
    :param label_key: Key for the labels
    :param cluster_key: Key for the clusters
    :return: Graph connectivity value
    """
    return metrics.graph_connectivity(adata, label_key)

def kbet(adata, label_key, batch_key):
    """
    Compute the kBET score between the labels and the batches.

    :param adata: Anndata object
    :param label_key: Key for the labels
    :param batch_key: Key for the batches
    :return: kBET score
    """
    try:
        M = metrics.kbet.diffusion_conn(adata, min_k=15, copy=False)
    except ValueError:
        # highly likely that the graph is not connected and the diffusion map cannot be computed
        return 0.0
    adata.obsp["connectivities"] = M

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', module='rpy2.robjects')
        warnings.filterwarnings('ignore', category=FutureWarning)
        kbet_score = metrics.kBET(
            adata,
            batch_key=batch_key,
            label_key=label_key,
            type_='knn',
            embed=None,
            scaled=True,
            verbose=False,
        )
    return kbet_score

def lisi_label(adata, label_key):
    clisi = metrics.clisi_graph(
        adata,
        label_key=label_key,
        type_='knn',
        subsample=100,  # Use all data
        scale=True,
        n_cores=8,
        verbose=False,
    )
    return clisi


def lisi_batch(adata, batch_key):
    ilisi = metrics.ilisi_graph(
        adata,
        batch_key=batch_key,
        type_='knn',
        subsample=100,  # Use all data
        scale=True,
        n_cores=8,
        verbose=False,
    )
    return ilisi

def asw(adata, label_key):
    feats = adata.X
    asw = silhouette_score(feats, adata.obs[label_key], metric='cosine')
    asw = (asw + 1) / 2
    return asw

def silhouette_batch(adata, label_key, batch_key):
    adata.obsm['X_embd'] = adata.X
    asw_batch = metrics.silhouette_batch(
        adata,
        batch_key,
        label_key,
        'X_embd',
        metric="cosine",
        verbose=False,
    )
    return asw_batch

def mean_average_precision_nonrep(adata):
    """
    Compute the mean average precision between the labels and the clusters.

    :param adata: Anndata object
    :param label_key: Key for the labels
    :return: mAP value
    """
    X = adata.X

    # compound == labels; plate/replicate == augmentation
    # mask query from same augmentation but different labels
    aps = []
    augmentations = adata.obs["augmentations"].cat.categories
    for aug in augmentations:
        mask = adata.obs["augmentations"] == aug
        X = adata.X[mask]

        distances = cdist(X, X, metric='cosine')
        ind = numpy.argsort(distances, axis=1)[:, 1:]

        labels = adata.obs["labels"].cat.codes[mask].values
        _, counts = numpy.unique(labels, return_counts=True)
        counts_per_labels = counts[labels]
        positives = labels[ind] == labels[:, numpy.newaxis]

        tp_at_k = numpy.cumsum(positives, axis=1)
        precision_at_k = tp_at_k / numpy.arange(1, tp_at_k.shape[1] + 1)
        recall_at_k = tp_at_k / counts_per_labels[:, numpy.newaxis]
        average_precision = numpy.sum((recall_at_k[:, 1:] - recall_at_k[:, :-1]) * precision_at_k[:, 1:], axis=1)

        aps.append(numpy.mean(average_precision))
    return numpy.mean(aps)

def get_data(features, labels, augmentations, categories=None):
    """
    Get the data for the experiment.

    :return: Anndata object
    """
    adata = anndata.AnnData(features)
    adata.obs_names = [f"image_{i}" for i in range(adata.n_obs)]
    adata.var_names = [f"feature_{i}" for i in range(adata.n_vars)]

    adata.obs["labels"] = pandas.Categorical(labels)
    adata.obs["augmentations"] = pandas.Categorical(augmentations, categories)

    # Preprocess the data
    scanpy.pp.neighbors(adata, use_rep='X', n_neighbors=25, metric='cosine')
    scanpy.tl.leiden(adata, key_added="cluster")

    return adata

def get_features(model, loader, batch_effects, device="cpu"):
    """
    Get the features from the model.

    :param model: Model
    :param loader: DataLoader
    :return: Anndata object
    """
    features, labels, augmentations = [], [], []
    
    model.eval()
    with torch.no_grad():

        for batch_effect in tqdm(batch_effects, desc="Batch effects"):
            for X, y in tqdm(loader, desc="Loader", leave=False):
                
                # Apply the batch effect
                X = batch_effect.apply(X)

                X = X.to(device)
                feats = model.forward_features(X)
                features.append(feats.cpu().numpy())
                labels.append(y["label"].cpu().numpy())
                augmentations.append([batch_effect.name] * X.size(0))

    features = numpy.concatenate(features, axis=0)
    labels = numpy.concatenate(labels, axis=0)
    augmentations = numpy.concatenate(augmentations, axis=0)
    return features, labels, augmentations

def get_scores(adata, label_key=None, batch_key=None, cluster_key=None):
    """
    Compute the batch correction metrics and the bio metrics.

    :param adata: Anndata object
    :param label_key: Key for the labels
    :param cluster_key: Key for the clusters
    :return: Dictionary of scores
    """
    scores = dict()
    scores["Graph-connectivity"] = graph_connectivity(adata.copy(), label_key)
    scores["KBET"] = kbet(adata.copy(), label_key, batch_key)
    scores["LISI-batch"] = lisi_batch(adata.copy(), batch_key)
    scores["Silhouette-batch"] = silhouette_batch(adata.copy(), label_key, batch_key)
    scores["LISI-label"] = lisi_label(adata.copy(), label_key)
    scores["Leiden-ARI"] = ari(adata.copy(), label_key, cluster_key)
    scores["Leiden-NMI"] = nmi(adata.copy(), label_key, cluster_key)
    scores["Silhouette-label"] = asw(adata.copy(), label_key)
    scores["mAP-nonrep"] = mean_average_precision_nonrep(adata.copy())
    return scores

def plot_embeddings(adata, effect, keys, mapper="phate"):
    """
    Calculate the embeddings and plot them.

    :param adata: Anndata object
    :param keys: List of keys

    :return: None
    """
    mappers = {
        "umap": UMAP,
        "phate": PHATE
    }
    mapper_op = mappers[mapper](random_state=args.seed)
    if mapper == "umap":
        X = adata.X
    else:
        X = adata
    transformed = mapper_op.fit_transform(X)

    for key in keys:
        fig, ax = pyplot.subplots(figsize=(8, 8))
        ax.scatter(transformed[:, 0], transformed[:, 1], c=adata.obs[key].cat.codes, s=10, cmap="nice-prism", rasterized=True)
        ax.axis("off")

        cmap = pyplot.get_cmap("nice-prism")
        norm = matplotlib.colors.Normalize(0, len(adata.obs[key].cat.categories)-1)
        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        ax.legend(
            handles=[
                matplotlib.lines.Line2D([], [], marker='o', color=m.to_rgba(i), markerfacecolor=m.to_rgba(i), markersize=5, linestyle="none", label=cat)
                for i, cat in enumerate(adata.obs[key].cat.categories)
            ],
            fontsize=8
        )
        os.makedirs(os.path.join("results", "embeddings"), exist_ok=True)
        savefig(fig, os.path.join("results", "embeddings", f"{args.dataset}_{args.model}_{args.weights}_{effect}_{key}"), extension="pdf", save_white=True)
        savefig(fig, os.path.join("results", "embeddings", f"{args.dataset}_{args.model}_{args.weights}_{effect}_{key}"), extension="png", save_white=True)
        pyplot.close(fig)

def show_examples(effect, images):
    """
    Show the images.

    :param images: List of images
    :return: None
    """
    arr_in = []
    for i in range(len(images)):
        vmin = min([image.min() for image in images[i]]).item()
        vmax = max([image.max() for image in images[i]]).item()
        for image in images[i]:
            image = image[0].numpy()
            arr_in.append(numpy.clip((image - vmin) / (vmax - vmin), 0, 1))
    output = montage(arr_in, grid_shape=(len(images), len(images[0])), padding_width=1, fill=min([im.min() for im in arr_in]))
    imsave(
        os.path.join("results", f"{args.dataset}_{effect}_examples.png"), 
        make_composite(output[numpy.newaxis], ["hot"])
    )

def main():

    set_seeds()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_channels = 3 if "imagenet" in args.weights.lower() else 1

    model, cfg = get_pretrained_model_v2(
        name=args.model,
        weights=args.weights,
        path=None,
        mask_ratio=0.0, 
        pretrained=True if n_channels==3 else False,
        in_channels=n_channels,
        as_classifier=True,
        blocks="all",
        num_classes=1
    )
    model = model.to(device)

    # Update configuration
    cfg.args = args
    update_cfg(cfg, args.opts)

    # Load the dataset
    train_loader, valid_loader, test_loader = get_dataset(
        name=args.dataset,
        transform=None,
        path=None,
        n_channels=n_channels,
        batch_size=cfg.batch_size,
        training=True
    )

    # histogram; 
    batch_effects = {
        "poisson" : [
            IdendityBatchEffect(),
            PoissonBatchEffect(1.0),
            PoissonBatchEffect(5.0),
            PoissonBatchEffect(10.0),
            PoissonBatchEffect(25.0),
        ],
        "gaussian" : [
            IdendityBatchEffect(),
            GaussianBatchEffect(0.0, 0.01),
            GaussianBatchEffect(0.0, 0.05),
            GaussianBatchEffect(0.0, 0.10),
        ],
        "contrast" : [
            IdendityBatchEffect(),
            ContrastBatchEffect(0.30, 10.0),
            ContrastBatchEffect(0.40, 10.0),
            ContrastBatchEffect(0.50, 10.0),
        ],
        "gaussian-blur" : [
            IdendityBatchEffect(),
            GaussianBlurBatchEffect(0.1),
            GaussianBlurBatchEffect(0.5),
            GaussianBlurBatchEffect(1.0),
            GaussianBlurBatchEffect(2.0),
        ],
    }

    # Save the results
    os.makedirs("results", exist_ok=True)
    scores = dict()
    for name, effects in batch_effects.items():

        images = []
        rng = numpy.random.default_rng(seed=args.seed)
        choices = rng.choice(len(train_loader.dataset), 5)
        for idx in choices:
            imgs = []
            for effect in effects:
                imgs.append(effect.apply(train_loader.dataset[idx][0]))
            images.append(imgs)
        show_examples(name, images)
        
        features, labels, augmentations = get_features(model, train_loader, batch_effects=effects, device=device)
        adata = get_data(features, labels, augmentations, categories=[effect.name for effect in effects])
        scores[name] = get_scores(adata, label_key="labels", batch_key="augmentations", cluster_key="cluster")
        plot_embeddings(adata, name, ["labels", "augmentations"])

    with open(os.path.join("results", f"{args.dataset}_{args.model}_{args.weights}.json"), "w") as f:
        json.dump(scores, f, indent=4, sort_keys=True)

if __name__ == "__main__":
    main()
