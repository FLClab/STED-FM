
import numpy
import pandas
import anndata
import warnings

from sklearn.metrics import silhouette_score

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')

    import scanpy
    from scib import metrics

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

def graph_connectivity(adata, label_key, cluster_key):
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
    M = metrics.kbet.diffusion_conn(adata, min_k=15, copy=False)
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

def get_data():
    """
    Get the data for the experiment.

    :return: Anndata object
    """
    augmentations, features = [], []
    for i in range(3):
        features.extend(numpy.random.normal(0, 0.1, size=(1000, 512)))
        augmentations.extend([f"augmentation-{i}"] * 1000)
    features, augmentations = numpy.array(features), numpy.array(augmentations)

    adata = anndata.AnnData(features)
    adata.obs_names = [f"cell_{i}" for i in range(adata.n_obs)]
    adata.var_names = [f"feature_{i}" for i in range(adata.n_vars)]
    adata.obs["label"] = pandas.Categorical(augmentations)

    # Preprocess the data
    scanpy.pp.neighbors(adata, use_rep='X', n_neighbors=25, metric='cosine')
    scanpy.tl.leiden(adata, key_added="cluster")

    return adata


def main():

    adata = get_data()

    print(adata)

    # Compute the batch correction metrics
    print("Graph connectivity:", graph_connectivity(adata, "label", "cluster"))
    # print("kbet:", kbet(adata, "label", "cluster"))
    print("LISI batch:", lisi_batch(adata, "cluster"))
    print("Silhouette batch:", silhouette_batch(adata, "label", "cluster"))

    # Compute the bio metrics
    print("LISI label:", lisi_label(adata, "label"))
    print("Leiden ARI:", ari(adata, "label", "cluster"))
    print("Leiden NMI:", nmi(adata, "label", "cluster"))
    print("Silhouette label:", asw(adata, "label"))



if __name__ == "__main__":
    main()
