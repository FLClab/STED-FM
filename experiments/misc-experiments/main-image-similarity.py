
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
from tqdm.auto import tqdm
from scipy import ndimage

from tiffwrapper import make_composite

from stedfm.loaders import get_dataset as get_classification_dataset
from stedfm.datasets import get_dataset, ArchiveDataset
from stedfm.datasets.segmentation import get_dataset as get_segmentation_dataset
from stedfm.DEFAULTS import BASE_PATH, COLORS, DATASETS
from stedfm.configuration import Configuration
from stedfm.utils import savefig

class DefaultConfiguration(Configuration):
    
    in_channels: int = 1

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
    return radialprofile[:power_spectrum.shape[0]//2]

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

    # Revert log transform
    profiles_a = 10**profiles_a
    profiles_b = 10**profiles_b

    # Normalize profiles
    profiles_a = profiles_a / (profiles_a.sum(axis=1, keepdims=True) + 1e-8)
    profiles_b = profiles_b / (profiles_b.sum(axis=1, keepdims=True) + 1e-8)

    if metric == "npsdd":
        # Normalized Power Spectrum Density Distance (NPSDD)
        # This corresponds to the area under the difference curve between two normalized power spectra
        distances = cdist(profiles_a, profiles_b, metric='minkowski', p=1.0)
    elif metric == "loglog-slope":
        log_profiles_a = numpy.log10(profiles_a + 1e-8)
        log_profiles_b = numpy.log10(profiles_b + 1e-8)
        distances = cdist(log_profiles_a, log_profiles_b, metric='correlation')
    else:
        distances = cdist(profiles_a, profiles_b, metric=metric)
    mask = numpy.triu(numpy.ones(distances.shape), k=1).astype(bool)
    return distances[mask].mean()

def compare_radial_profiles(files, names, metric="euclidean"):
    distances = numpy.zeros((len(files), len(files)))
    # for file_a, file_b in tqdm(itertools.combinations(files, 2), total=len(files)*(len(files)-1)//2):
    for i, j in tqdm(itertools.combinations(range(len(files)), 2), total=len(files)*(len(files)-1)//2):
        file_a = files[i]
        file_b = files[j]
        # if not (names[i] in ["STED", "SIM", "HPA", "JUMP", "ImageNet"]):
        #     continue

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

def plot_distance_heatmap(distances, names, savename='radial_profile_distances'):
    fig, ax = pyplot.subplots()
    cax = ax.imshow(distances, cmap='RdPu')
    fig.colorbar(cax)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90)
    ax.set_yticklabels(names)
    pyplot.tight_layout()
    savefig(fig, f"./figures/image-similarity/{savename}")

def plot_graphs(distances, labels, savename='radial_profile_graph'):
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
    savefig(fig, f"./figures/image-similarity/{savename}")

def plot_mds(distances, labels, savename='radial_profile_mds'):
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
    savefig(fig, f"./figures/image-similarity/{savename}")

def compute_fractal_dimensions(dataset, num_samples=5000, crop_size=224):
    """Compute fractal dimensions for a subset of images in the dataset."""
    numpy.random.seed(42)
    indices = numpy.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    fds = []
    for idx in indices:
        image = dataset[idx]
        if isinstance(image, (list, tuple)):
            image = image[0]
        image = image.numpy().squeeze()

        image = crop_center(image, crop_size=crop_size)

        sobel_h = ndimage.sobel(image, 0)  # horizontal gradient
        sobel_v = ndimage.sobel(image, 1)  # vertical gradient
        magnitude = numpy.sqrt(sobel_h**2 + sobel_v**2)
        threshold = numpy.percentile(magnitude, 75)
        binary_image = magnitude > threshold

        # fig, ax = pyplot.subplots()
        # composite = make_composite(numpy.stack([image, binary_image.astype(numpy.float32)]), luts=['gray', 'green'], ranges=[(0, 1), (0, 2)])
        # ax.imshow(composite)
        # ax.axis('off')
        # savefig(fig, f"./figures/image-similarity/fractal_dimension_{idx}", dpi=300)
        # pyplot.close(fig)

        # Box-counting method
        sizes = 2**numpy.arange(1, int(numpy.log2(crop_size))+1)
        counts = []
        for size in sizes:
            S = numpy.add.reduceat(
                numpy.add.reduceat(binary_image, numpy.arange(0, binary_image.shape[0], size), axis=0),
                numpy.arange(0, binary_image.shape[1], size), axis=1)
            counts.append(numpy.sum(S > 0)) # Count non-empty boxes
        counts = numpy.array(counts)

        # Linear fit in log-log space
        log_sizes = numpy.log(1 / sizes)
        log_counts = numpy.log(counts + 1e-8)  # Avoid log(0)
        coeffs = numpy.polyfit(log_sizes, log_counts, 1)
        fd = coeffs[0]
        fds.append(fd)

        # fig, ax = pyplot.subplots()
        # ax.plot(log_sizes, log_counts, 'o', label='Data')
        # ax.plot(log_sizes, numpy.polyval(coeffs, log_sizes), '-', label=f'Fit (FD={fd:.2f})')
        # ax.set_xlabel('log(1/box size)')
        # ax.set_ylabel('log(box count)') 
        # ax.legend()
        # savefig(fig, f"./figures/image-similarity/fractal_dimension_fit_{idx}", dpi=300)
        # pyplot.close(fig)

    return fds

def get_fractal_dimensions(files, names):
    fractal_dimensions = {}
    for name, profile_file in zip(names, files):
        profile_data = numpy.load(profile_file)
        fractal_dimensions[name] = profile_data
    return fractal_dimensions

def plot_fractal_dimensions(fractal_dimensions, names):
    fig, ax = pyplot.subplots(figsize=(4, 3))
    for name in names:
        data = fractal_dimensions[name]
        ax.hist(data, range=(1.5, 2.0), bins=100, alpha=0.5, label=f"{name} (mean={data.mean():.2f})", color=COLORS[name])
    ax.set_xlabel("Fractal Dimension")
    ax.set_ylabel("Frequency")
    # ax.legend()
    savefig(fig, "./figures/image-similarity/fractal_dimensions", dpi=300)

def compare_fractal_dimensions(fractal_dimensions, names, metric="euclidean"):
    distances = numpy.zeros((len(names), len(names)))
    for i, j in tqdm(itertools.combinations(range(len(names)), 2), total=len(names)*(len(names)-1)//2):
        name_a = names[i]
        name_b = names[j]
        fd_a = fractal_dimensions[name_a]
        fd_b = fractal_dimensions[name_b]

        distance = wasserstein_distance(fd_a, fd_b)

        distances[i, j] = distance
        distances[j, i] = distance
    return distances

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="STED")
    parser.add_argument("--dataset-path", type=str, default=f"{BASE_PATH}/ssl-data/")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--metric", type=str, default="npsdd", choices=["euclidean", "cosine", "correlation", "npsdd", "loglog-slope"])
    parser.add_argument("--measure", type=str, default="radial-profile", choices=["all", "radial-profile", "fractal-dimension"])
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
        distances = compare_radial_profiles(files, names, metric=args.metric)
        numpy.savez("results/radial_profile_distances.npz", distances=distances, names=names)
        plot_distance_heatmap(distances, names)
        plot_graphs(distances, names)
        plot_mds(distances, names)

        files = glob.glob("results/fractal-dimensions/*.npy")
        files = [
            "results/fractal-dimensions/STED.npy",
            "results/fractal-dimensions/SIM.npy",
            "results/fractal-dimensions/HPA.npy",
            "results/fractal-dimensions/JUMP.npy",
            "results/fractal-dimensions/ImageNet.npy",
            "results/fractal-dimensions/optim.npy",
            "results/fractal-dimensions/neural-activity-states.npy",
            "results/fractal-dimensions/peroxisome.npy",
            "results/fractal-dimensions/polymer-rings.npy",
            "results/fractal-dimensions/dl-sim.npy",
            "results/fractal-dimensions/bbbc026.npy",
            "results/fractal-dimensions/bbbc052.npy",
            "results/fractal-dimensions/bbbc053.npy",
            "results/fractal-dimensions/factin.npy",
            "results/fractal-dimensions/footprocess.npy",
            "results/fractal-dimensions/lioness.npy",
            "results/fractal-dimensions/synaptic-semantic-segmentation.npy",
            "results/fractal-dimensions/lcn.npy",
            "results/fractal-dimensions/deepd3.npy",
        ]
        names = [os.path.basename(f).replace(".npy", "") for f in files]
        names = [DATASETS[name] for name in names]        

        fractal_dimensions = get_fractal_dimensions(files, names)
        plot_fractal_dimensions(fractal_dimensions, names)
        distances = compare_fractal_dimensions(fractal_dimensions, names)
        numpy.savez("results/fractal_dimension_distances.npz", distances=distances, names=names)
        plot_distance_heatmap(distances, names, savename='fractal_dimension_distances')
        plot_graphs(distances, names, savename='fractal_dimension_graph')
        plot_mds(distances, names, savename='fractal_dimension_mds')
        return
    

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

    # Segmentation datasets
    elif args.dataset in ["factin", "footprocess", "lioness", "synaptic-semantic-segmentation", "lcn", "deepd3"]:
        dataset, _, _ = get_segmentation_dataset(
            name=args.dataset,
            cfg=DefaultConfiguration(),
            split="train",
            use_cache=False
        )
    
    # Classification datasets
    elif args.dataset in ["optim", "neural-activity-states", "peroxisome", "polymer-rings", "dl-sim", "bbbc026", "bbbc052", "bbbc053"]:
        loader, _, _ = get_classification_dataset(
            name=args.dataset,
            cfg=DefaultConfiguration(),
            split="train",
            use_cache=False
        )
        dataset = loader.dataset
    else:
        raise NotImplementedError(f"`{args.dataset}` is not a valid option.")

    if args.measure == "all" or args.measure == "radial-profile":
        print(f"Computing radial profiles for dataset: {args.dataset}")
        profiles = compute_radial_profiles(dataset, num_samples=args.num_samples, crop_size=args.crop_size)
        os.makedirs("results/radial_profiles", exist_ok=True)
        numpy.save(f"results/radial_profiles/{args.dataset}.npy", profiles)
    if args.measure == "all" or args.measure == "fractal-dimension":
        print(f"Computing Fractal Dimensions for dataset: {args.dataset}")
        fds = compute_fractal_dimensions(dataset)
        fds = numpy.array(fds)
        os.makedirs("results/fractal-dimensions", exist_ok=True)
        numpy.save(f"results/fractal-dimensions/{args.dataset}.npy", fds)
        print(f"Mean Fractal Dimension: {fds.mean():.4f} ± {fds.std():.4f}")

if __name__ == "__main__":
    main()