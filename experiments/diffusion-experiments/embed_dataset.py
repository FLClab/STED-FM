import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import random 
import json 
from tqdm import tqdm, trange
import argparse 
from attribute_datasets import OptimQualityDataset, ProteinActivityDataset, LowHighResolutionDataset, TubulinActinDataset, ALSDataset
import os
from torch.utils.data import DataLoader

import numpy
from wavelet import detect_spots
from skimage import measure, feature
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter

import sys 
sys.path.insert(0, "../")
from utils import set_seeds
from DEFAULTS import BASE_PATH
from model_builder import get_pretrained_model_v2
from datasets import MICRANetHDF5Dataset, FactinCaMKIIDataset

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--blocks", type=str, default="all")
parser.add_argument("--global-pool", type=str, default="avg")
parser.add_argument("--num-per-class", type=int, default=None)
parser.add_argument("--precomputed", action="store_true")
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--dataset", type=str, default="quality")
args = parser.parse_args()

def load_dataset(balance=True) -> torch.utils.data.Dataset: 
    if args.dataset == "quality":
        dataset = OptimQualityDataset(
            data_folder=os.path.join(BASE_PATH, f"evaluation-data/optim_{args.split}"),
            num_samples={"actin": None},
            apply_filter=True,
            classes=['actin'],
            high_score_threshold=0.70,
            low_score_threshold=0.60,
            n_channels=3 if "imagenet" in args.weights.lower() else 1
        )
    elif args.dataset == "activity":
        dataset = ProteinActivityDataset(
            tarpath=os.path.join(BASE_PATH, f"evaluation-data/NeuralActivityStates/NAS_PSD95_{args.split}_v2.tar"),
            num_samples=None,
            transform=None,
            n_channels=3 if "imagenet" in args.weights.lower() else 1,
            num_classes=2,
            # protein_id=3,
            balance=balance,
            classes=["Block", "0MgGlyBic"]
        )
    elif args.dataset == "activity-block-glugly":
        dataset = ProteinActivityDataset(
            tarpath=os.path.join(BASE_PATH, f"evaluation-data/NeuralActivityStates/NAS_PSD95_{args.split}_v2.tar"),
            num_samples=None,
            transform=None,
            n_channels=3 if "imagenet" in args.weights.lower() else 1,
            num_classes=2,
            # protein_id=3,
            balance=balance,
            classes=["Block", "GluGly"]
        )        
    elif args.dataset == "resolution":
        path = os.path.join(BASE_PATH, "evaluation-data/low-high-quality")
        dataset = LowHighResolutionDataset(
            h5path=f"{path}/{args.split}.hdf5",
            num_samples=None,
            transform=None,
            n_channels=3 if "imagenet" in args.weights.lower() else 1,
            num_classes=2,
            classes=["low", "high"] 
        )
    elif args.dataset == "tubulin-actin":
        path = os.path.join(BASE_PATH, f"evaluation-data/optim_{args.split}")
        dataset = TubulinActinDataset(
            data_folder=path,
            classes=["tubulin", "actin"],
            n_channels=3 if "imagenet" in args.weights.lower() else 1,
            min_quality_score=0.70
        )
    elif args.dataset == "factin-rings-fibers":
        dataset = MICRANetHDF5Dataset(
            os.path.join(BASE_PATH, "evaluation-data", "actin-data", f"{args.split}_01-04-19.hdf5"),
            validation=True,
            n_channels=3 if "imagenet" in args.weights.lower() else 1,
            return_non_ambiguous=True,
            size=224
        )
    elif args.dataset == "factin-camkii":
        dataset = FactinCaMKIIDataset(
            os.path.join(BASE_PATH, "evaluation-data", "factin-camkii", f"{args.split}-dataset.tar"),
            n_channels=3 if "imagenet" in args.weights.lower() else 1,
            balance=True,
            classes=["CTRL", "shRNA"]
        )
    elif args.dataset == "factin-camkii-rescue":
        dataset = FactinCaMKIIDataset(
            os.path.join(BASE_PATH, "evaluation-data", "factin-camkii", f"{args.split}-dataset.tar"),
            n_channels=3 if "imagenet" in args.weights.lower() else 1,
            balance=False,
            classes=["CTRL", "RESCUE"]
        )        
    elif args.dataset == "als":
        dataset = ALSDataset(
            tarpath=f"/home-local/Frederic/Datasets/ALS/catalog/PLKO_{args.split}.tar",
        )
    else:
        raise ValueError(f"Dataset {args.dataset} not found")
    return dataset

def extract_features(image, show=False):
        
    image = image.cpu().numpy()
    image = image[0]

    filtered_image = gaussian_filter(image, sigma=1.0)

    PIXELSIZE = 0.015 # in um
    # mask = detect_spots(image, J_list=[2, 3], scale_threshold=5.0)
    mask = detect_spots(image, J_list=[3, 4], scale_threshold=2.0)

    foreground = np.count_nonzero(mask)
    pixels = image.shape[0] * image.shape[1]
    ratio = foreground / pixels 
    threshold = 0.06 if args.dataset == "als" else 0.05
    if ratio < threshold:
        return None

    mask_label, num_proteins = measure.label(mask, return_num=True)
    props = measure.regionprops(mask_label, intensity_image=image)
    coordinates = numpy.array([p.weighted_centroid for p in props])

    if len(coordinates) < 2:
        return None

    distances = distance.pdist(coordinates) * PIXELSIZE
    distances = distance.squareform(distances)

    nn_distances = numpy.sort(distances, axis=1)[:, 1]

    image_density = num_proteins / (image.shape[0] * image.shape[1] * PIXELSIZE**2)
    density_proteins = (numpy.sum(distances < 0.5, axis=1) - 1) / 1 # in number of proteins per um^2

    features = []
    counter = 0
    for prop, density, nn in zip(props, density_proteins, nn_distances):

        # img = prop.intensity_image
        slc = prop.slice
        img = filtered_image[slc]
        label = prop.image

        min_distance = int(0.08 / PIXELSIZE) // 2 + 1 # in pixels
        peaks = feature.peak_local_max(img, min_distance=min_distance, exclude_border=False, labels=label)

        features.append([
            prop.area,
            prop.perimeter,
            prop.mean_intensity,
            prop.eccentricity,
            prop.solidity,
            density,
            nn,
            len(peaks)
        ])

        # counter += 1
        # if counter > 5:
        #     break

    return numpy.array(features)   

if __name__=="__main__":

    set_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    model, cfg = get_pretrained_model_v2(
        name=args.model,
        weights=args.weights,
        path=None,
        mask_ratio=0.0,
        pretrained=True if "imagenet" in args.weights.lower() else False,
        in_channels=3 if "imagenet" in args.weights.lower() else 1,
        as_classifier=True, 
        blocks=args.blocks,
        num_classes=2
    )
    model = model.to(device)

    # dataset = load_dataset(balance=False)
    # exit()

    # print(f"Dataset size: {len(dataset)}")
    # print(np.unique(dataset.labels, return_counts=True))
    
    # from collections import defaultdict
    # aggregated = defaultdict(list)
    # for i in trange(len(dataset)):
    #     image, metadata = dataset[i]
    #     features = extract_features(image, show=False)
    #     if features is None:
    #         continue
    #     aggregated[metadata["condition"]].append(features)
    
    # import pickle
    # os.makedirs(f"./{args.dataset}-experiment/features", exist_ok=True)
    # with open(f"./{args.dataset}-experiment/features/real-images-features-{args.split}.pkl", "wb") as f:
    #     pickle.dump(aggregated, f)
    
    # exit()
    dataset = load_dataset()

    print(f"Dataset size: {len(dataset)}")
    # print(np.unique(dataset.labels, return_counts=True))    

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    embeddings = []
    all_labels = []
    model.eval()
    dataset = dataloader.dataset 
    N = len(dataset)
    with torch.no_grad():
        for i in range(N):
            images, data_dict = dataset[i]
            masks = detect_spots(images.squeeze().cpu().numpy(), J_list=[3, 4], scale_threshold=2.0)
            foreground = np.count_nonzero(masks)
            pixels = images.shape[1] * images.shape[2]
            ratio = foreground / pixels 
            threshold = 0.06 if args.dataset == "als" else 0.05
            if ratio < threshold:
                # print(f"Skipping {i} because ratio is {ratio} < {threshold}")
                continue

            # fig, axs = plt.subplots(1, 2)
            # axs[0].imshow(images.squeeze().cpu().numpy())
            # axs[1].imshow(masks)
            # fig.savefig(f"./{args.dataset}-experiment/temporary/example-{i}.png", dpi=1200, bbox_inches="tight")
            # plt.close(fig)

            images = images.unsqueeze(0).to(device)

            if "condition" in data_dict:
                labels = data_dict["condition"]
            else:
                labels = data_dict["label"]
                # print(labels)
            
            features = model.forward_features(images)
            embeddings.append(features.cpu().detach().numpy())
            if args.dataset == "als":
                all_labels.append(labels)
            else:
                all_labels.extend(labels)

    embeddings = np.concatenate(embeddings, axis=0)
    print(embeddings.shape)
    all_labels = np.array(all_labels)
    print(all_labels.shape)

    # Make sure labels are unique and in increasing order
    unique_labels = np.unique(all_labels)
    tmp = np.zeros_like(all_labels)
    labels_mapping = {}
    for i, label in enumerate(unique_labels):
        idx = np.where(all_labels == label)[0]
        tmp[idx] = i
        labels_mapping[label] = i
    all_labels = tmp

    print(np.unique(all_labels, return_counts=True))
    os.makedirs(f"./{args.dataset}-experiment/embeddings", exist_ok=True)
    with open(f"./{args.dataset}-experiment/embeddings/{args.weights}-{args.dataset}-labels_{args.split}.json", "w") as f:
        json.dump(labels_mapping, f)
    np.savez(f"./{args.dataset}-experiment/embeddings/{args.weights}-{args.dataset}-embeddings_{args.split}.npz", embeddings=embeddings, labels=all_labels)

