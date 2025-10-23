
import os
import io
import glob
import numpy
import torch
import itertools
import networkx

from matplotlib import pyplot
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
from PIL import Image
from torchvision.transforms import Resize

from stedfm.loaders import get_dataset as get_classification_dataset
from stedfm.datasets import get_dataset, ArchiveDataset
from stedfm.DEFAULTS import BASE_PATH, COLORS, DATASETS
from stedfm.configuration import Configuration
from stedfm.utils import savefig

class DefaultConfiguration(Configuration):
    
    in_channels: int = 1

import sys
sys.path.insert(0, "../segmentation-experiments")
from datasets import get_dataset as get_segmentation_dataset

DATASETS.factin = "F-Actin"
DATASETS.lioness = "Lioness"
DATASETS.peroxisome = "Px"
DATASETS.footprocess = "FP"
DATASETS.polymer_rings = "PR"
DATASETS.dl_sim = "DL-SIM"
DATASETS.neural_activity_states = "NAS"
DATASETS.optim = "SO"
DATASETS.bbbc026 = "BBBC026"
DATASETS.bbbc052 = "BBBC052"
DATASETS.bbbc053 = "BBBC053"
DATASETS.lcn = "LCN"
DATASETS.deepd3 = "DeepD3"
DATASETS.synaptic_semantic_segmentation = "SPZ"

DATASETSPATH = {
    "JUMP" : "jump.tar",
    "HPA" : "hpa.zip",
    "SIM" : "sim-dataset-crops.tar",
    "STED" : "STED-FM-dataset-crops.tar",
    "ImageNet" : "ILSVRC2012_img_test_v10102019.tar"
}

class ImageNetDataset(ArchiveDataset):
    """Placeholder for ImageNet dataset."""
    def __init__(self, path, use_cache=False, *args, **kwargs):
        
        self.debug = False

        super().__init__(path, use_cache=use_cache, *args, **kwargs)
        # Implement ImageNet specific loading if needed


    def get_members(self):
        if self.debug:
            members = [self.get_reader().next() for _ in range(5000)]
            return list(sorted(members, key=lambda m: m.name))           
        return list(sorted(self.get_reader().getmembers(), key=lambda m: m.name))
    
    def get_item_from_archive(self, member):
        # Loads file from TarFile stored as numpy array
        buffer = io.BytesIO()
        buffer.write(self.get_reader().extractfile(member).read())
        buffer.seek(0)        

        data = Image.open(buffer).convert("L")  # Convert to grayscale
        data = numpy.array(data)
        data = torch.from_numpy(data).unsqueeze(0).float()  # Add channel dimension

        data = Resize((224, 224))(data)
        return data

def crop_center(image, crop_size):
    """Crop the center of a 2D image."""
    y, x = image.shape
    startx = x//2 - (crop_size//2)
    starty = y//2 - (crop_size//2)    
    return image[starty:starty+crop_size, startx:startx+crop_size]

def hamming_window(size):
    """Generate a 2D Hamming window."""
    hamming_1d = numpy.hamming(size)
    hamming_2d = numpy.outer(hamming_1d, hamming_1d)
    return hamming_2d

def get_power_spectrum(image):
    """Compute the power spectrum of an image."""

    # Apply Hamming window to reduce edge effects
    hamming = hamming_window(image.shape[0])
    image = image * hamming

    f = numpy.fft.fft2(image)
    fshift = numpy.fft.fftshift(f)
    magnitude_spectrum = numpy.abs(fshift)
    power_spectrum = magnitude_spectrum**2
    return power_spectrum

def radial_profile(power_spectrum):
    """Compute the radial profile of a 2D power spectrum."""
    y, x = numpy.indices(power_spectrum.shape)
    center = numpy.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    r = numpy.hypot(x - center[0], y - center[1])
    r = r.astype(int)

    tbin = numpy.bincount(r.ravel(), power_spectrum.ravel())
    nr = numpy.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def compute_radial_profiles(dataset, num_samples=5000, crop_size=224):
    """Compute radial profiles for a subset of images in the dataset."""
    numpy.random.seed(42)
    indices = numpy.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    profiles = []
    for idx in indices:
        image = dataset[idx]
        if isinstance(image, (list, tuple)):
            image = image[0]
        image = image.numpy().squeeze()

        image = crop_center(image, crop_size=crop_size)
        power_spectrum = get_power_spectrum(image)
        radial_prof = radial_profile(power_spectrum)

        profiles.append(numpy.log10(radial_prof))
    
    return numpy.array(profiles)

def compute_distance(profiles_a, profiles_b, metric="euclidean"):
    """Compute distance between two parameter vectors."""
    # if metric == "wasserstein":
    #     return wasserstein_distance(profiles_a, profiles_b)
    # else:
    if len(profiles_a) > len(profiles_b):
        profiles_a, profiles_b = profiles_b, profiles_a
    distances = cdist(profiles_a, profiles_b, metric=metric)
    mask = numpy.triu(numpy.ones(distances.shape), k=1).astype(bool)
    return distances[mask].mean()

def compare_radial_profiles(files, metric="euclidean"):
    distances = numpy.zeros((len(files), len(files)))
    for file_a, file_b in itertools.combinations(files, 2):

        profiles_a = numpy.load(file_a)
        profiles_b = numpy.load(file_b)

        distance = compute_distance(profiles_a, profiles_b, metric=metric)

        distances[files.index(file_a), files.index(file_b)] = distance
        distances[files.index(file_b), files.index(file_a)] = distance
    return distances

def plot_radial_profiles(profiles, names, num_examples=15):
    for name, profile_file in zip(names, profiles):
        profile_data = numpy.load(profile_file)

        fig, ax = pyplot.subplots(figsize=(4, 3))
        random_indices = numpy.random.choice(profile_data.shape[0], size=min(num_examples, profile_data.shape[0]), replace=False)
        for i in random_indices:
            ax.plot(profile_data[i], alpha=0.5, color='silver')
        
        mean, std = profile_data.mean(axis=0), profile_data.std(axis=0)
        ax.plot(profile_data.mean(axis=0), color='black', linewidth=2, label='Mean Profile')
        ax.fill_between(numpy.arange(len(mean)), mean - std, mean + std, alpha=0.3)

        ax.set_xlabel("Spatial Frequency")
        ax.set_ylabel("Log Power")
        ax.set_title(f"{name} (n={profile_data.shape[0]})")

        savefig(fig, f"./figures/image-similarity/radial_profiles_{name}", dpi=300)
        pyplot.close(fig)

def plot_distance_heatmap(distances, names):
    fig, ax = pyplot.subplots()
    cax = ax.imshow(distances, cmap='RdPu')
    fig.colorbar(cax)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90)
    ax.set_yticklabels(names)
    pyplot.tight_layout()
    savefig(fig, "./figures/image-similarity/radial_profile_distances", dpi=300)

def plot_graphs(distances, labels):
    G = networkx.Graph()

    distances = (distances - distances.min()) / (distances.max() - distances.min())
    for j in range(len(labels)):
        for i in range(j):
            if i != j:
                G.add_edge(labels[i], labels[j], weight=distances[i, j])
    
    fig, ax = pyplot.subplots()
    networkx.draw_spring(
        G, with_labels=True, ax=ax,
        node_color=[COLORS[node] for node in G.nodes],
        font_size=10,
        width=1.0
    )
    savefig(fig, "./figures/image-similarity/graph")

def plot_mds(distances, labels):
    from sklearn.manifold import MDS

    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(distances)

    fig, ax = pyplot.subplots()
    point_colors = [COLORS[label] for label in labels]
    ax.scatter(coords[:, 0], coords[:, 1], c=point_colors, s=100)
    ax.set_aspect('equal')
    for i, label in enumerate(labels):
        ax.annotate(label, (coords[i, 0], coords[i, 1]), color="black", horizontalalignment='center', verticalalignment='center')
    ax.set_axis_off()
    savefig(fig, "./figures/image-similarity/mds")

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="STED")
    parser.add_argument("--dataset-path", type=str, default=f"{BASE_PATH}/ssl-data/")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--plot", action='store_true', help="Whether to plot example images and spectra.")
    args = parser.parse_args()

    if args.plot:
        files = glob.glob("results/radial_profiles/*.npy")
        files = [
            "results/radial_profiles/STED.npy",
            "results/radial_profiles/SIM.npy",
            "results/radial_profiles/HPA.npy",
            "results/radial_profiles/JUMP.npy",
            "results/radial_profiles/ImageNet.npy",
            "results/radial_profiles/optim.npy",
            "results/radial_profiles/neural-activity-states.npy",
            "results/radial_profiles/peroxisome.npy",
            "results/radial_profiles/polymer-rings.npy",
            "results/radial_profiles/dl-sim.npy",
            "results/radial_profiles/bbbc026.npy",
            "results/radial_profiles/bbbc052.npy",
            "results/radial_profiles/bbbc053.npy",
            "results/radial_profiles/factin.npy",
            "results/radial_profiles/footprocess.npy",
            "results/radial_profiles/lioness.npy",
            "results/radial_profiles/synaptic-semantic-segmentation.npy",
            "results/radial_profiles/lcn.npy",
            "results/radial_profiles/deepd3.npy",
        ]
        names = [os.path.basename(f).replace(".npy", "") for f in files]
        names = [DATASETS[name] for name in names]
        plot_radial_profiles(files, names)

        distances = compare_radial_profiles(files, metric="correlation")

        numpy.savez("results/radial_profile_distances.npz", distances=distances, names=names)
        
        plot_distance_heatmap(distances, names)
        plot_graphs(distances, names)
        plot_mds(distances, names)

        return
    
    print(f"Computing radial profiles for dataset: {args.dataset}")

    # Pretraining datasets
    if args.dataset in ["JUMP", "HPA", "SIM", "STED", "ImageNet"]:
        if args.dataset == "ImageNet":
            dataset = ImageNetDataset(
                path=f"{args.dataset_path}/{DATASETSPATH[args.dataset]}",
                use_cache=False
            )
        else:
            dataset = get_dataset(
                args.dataset,
                path=f"{args.dataset_path}/{DATASETSPATH[args.dataset]}",
                use_cache=False
            )
        profiles = compute_radial_profiles(dataset, num_samples=args.num_samples, crop_size=args.crop_size)

        os.makedirs("results/radial_profiles", exist_ok=True)
        numpy.save(f"results/radial_profiles/{args.dataset}.npy", profiles)

    # Segmentation datasets
    elif args.dataset in ["factin", "footprocess", "lioness", "synaptic-semantic-segmentation", "lcn", "deepd3"]:
        dataset, _, _ = get_segmentation_dataset(
            name=args.dataset,
            cfg=DefaultConfiguration(),
            split="train",
            use_cache=False
        )
        profiles = compute_radial_profiles(dataset, num_samples=args.num_samples, crop_size=args.crop_size)

        os.makedirs("results/radial_profiles", exist_ok=True)
        numpy.save(f"results/radial_profiles/{args.dataset}.npy", profiles)
    
    # Classification datasets
    elif args.dataset in ["optim", "neural-activity-states", "peroxisome", "polymer-rings", "dl-sim", "bbbc026", "bbbc052", "bbbc053"]:
        loader, _, _ = get_classification_dataset(
            name=args.dataset,
            cfg=DefaultConfiguration(),
            split="train",
            use_cache=False
        )
        dataset = loader.dataset
        profiles = compute_radial_profiles(dataset, num_samples=args.num_samples, crop_size=args.crop_size)

        os.makedirs("results/radial_profiles", exist_ok=True)
        numpy.save(f"results/radial_profiles/{args.dataset}.npy", profiles)
    
    else:
        raise NotImplementedError(f"`{args.dataset}` is not a valid option.")

if __name__ == "__main__":
    main()