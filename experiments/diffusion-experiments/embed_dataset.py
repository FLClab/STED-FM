import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import random 
import json 
from tqdm import tqdm, trange
import argparse 
from attribute_datasets import OptimQualityDataset, ProteinActivityDataset, LowHighResolutionDataset, TubulinActinDataset
import os
from torch.utils.data import DataLoader

import numpy
from wavelet import detect_spots
from skimage import measure, feature
from scipy.spatial import distance

import sys 
sys.path.insert(0, "../")
from utils import set_seeds
from DEFAULTS import BASE_PATH
from model_builder import get_pretrained_model_v2

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

def load_dataset() -> torch.utils.data.Dataset: 
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
            balance=True,
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
            balance=True,
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
    else:
        raise ValueError(f"Dataset {args.dataset} not found")
    return dataset


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

    dataset = load_dataset()

    print(f"Dataset size: {len(dataset)}")
    print(np.unique(dataset.labels, return_counts=True))
    
    def extract_features(image, show=False):
        
        image = image.cpu().numpy()
        image = image[0]

        PIXELSIZE = 0.015 # in um
        # mask = detect_spots(image, J_list=[2, 3], scale_threshold=5.0)
        mask = detect_spots(image, J_list=[3, 4], scale_threshold=2.0)

        if show:    
            fig, axes = pyplot.subplots(1, 2)
            axes[0].imshow(image, cmap='gray')
            axes[1].imshow(mask, cmap='gray')
            pyplot.show()

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

            img = prop.intensity_image
            label = prop.image

            min_distance = int(0.08 / PIXELSIZE) // 2 + 1 # in pixels
            peaks = feature.peak_local_max(img, min_distance=min_distance, exclude_border=False)

            if show:
                fig, axes = pyplot.subplots(1, 2)
                axes[0].imshow(img, cmap='gray')
                axes[0].plot(peaks[:, 1], peaks[:, 0], 'r.')
                axes[1].imshow(label, cmap='gray')
                pyplot.show()

            features.append([
                prop.area,
                prop.perimeter,
                prop.mean_intensity,
                prop.eccentricity,
                prop.solidity,
                density,
                nn,
                # len(peaks)
            ])

            # counter += 1
            # if counter > 5:
            #     break

        return numpy.array(features)    

    # from collections import defaultdict
    # aggregated = defaultdict(list)
    # for i in trange(len(dataset)):
    #     image, metadata = dataset[i]
    #     features = extract_features(image, show=False)
    #     if features is None:
    #         continue
    #     aggregated[metadata["condition"]].append(features)
    
    # import pickle
    # with open("./data.pkl", "wb") as f:
    #     pickle.dump(aggregated, f)
    
    # exit()

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    embeddings = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, data_dict = batch
            images = images.to(device)
            labels = data_dict["condition"]
            features = model.forward_features(images)
            embeddings.append(features.cpu().detach().numpy())
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

    os.makedirs(f"./{args.dataset}-experiment/embeddings", exist_ok=True)
    with open(f"./{args.dataset}-experiment/embeddings/{args.weights}-{args.dataset}-labels_{args.split}.json", "w") as f:
        json.dump(labels_mapping, f)
    np.savez(f"./{args.dataset}-experiment/embeddings/{args.weights}-{args.dataset}-embeddings_{args.split}.npz", embeddings=embeddings, labels=all_labels)

