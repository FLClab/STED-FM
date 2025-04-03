import json 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize 
from wavelet import detect_spots 
import argparse 
import torch 
from torch import nn 
from diffusion_models.diffusion.ddpm_lightning import DDPM 
from diffusion_models.diffusion.denoising.unet import UNet 
from tqdm import trange, tqdm 
import copy 
import random 
import os 
from skimage import measure 
from typing import Union, List 
from attribute_datasets import ALSDataset  
import sys 
from scipy.spatial.distance import cdist 
import glob 
from PIL import Image
import pickle 
import tifffile 
sys.path.insert(0, "../")
from DEFAULTS import BASE_PATH, COLORS 
from model_builder import get_pretrained_model_v2 
from utils import set_seeds 


parser = argparse.ArgumentParser()
parser.add_argument("--latent-encoder", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--timesteps", type=int, default=1000)
parser.add_argument("--boundary", type=str, default="als")
parser.add_argument("--num-samples", type=int, default=20)
parser.add_argument("--ckpt-path", type=str, default=f"{BASE_PATH}/baselines/DiffusionModels/latent-guidance")
parser.add_argument("--figure", action="store_true")
parser.add_argument("--sanity-check", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--direction", type=str, default="DIV14")
parser.add_argument("--n-steps", type=int, default=5)
args = parser.parse_args()

def linear_interpolate(latent_code,
                       boundary,
                       intercept,
                       start_distance=-4.0,
                       end_distance=4.0,
                       steps=8):
  assert (latent_code.shape[0] == 1 and boundary.shape[0] == 1 and
          len(boundary.shape) == 2 and
          boundary.shape[1] == latent_code.shape[-1])

  linspace = np.linspace(start_distance, end_distance, steps)
  if len(latent_code.shape) == 2:
    linspace = linspace - (latent_code.dot(boundary.T) + intercept)
    linspace = linspace.reshape(-1, 1).astype(np.float32)
    return latent_code + linspace * boundary, linspace
  if len(latent_code.shape) == 3:
    linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
    return latent_code + linspace * boundary.reshape(1, 1, -1), linspace
  raise ValueError(f'Input `latent_code` should be with shape '
                   f'[1, latent_space_dim] but {latent_code.shape} was received.')

def load_svm():
    with open(f"./{args.boundary}-experiment/boundaries/{args.weights}_{args.boundary}_svm.pkl", "rb") as f:
        return pickle.load(f)

def load_boundary() -> np.ndarray:
    print(f"--- Loading boundary trained from {args.weights} embeddings ---")
    data = np.load(f"./{args.boundary}-experiment/boundaries/{args.weights}_{args.boundary}_boundary.npz")
    boundary, intercept, norm = data["boundary"], data["intercept"], data["norm"]
    return boundary, intercept, norm

def extract_features(img: Union[np.ndarray, torch.Tensor], check_foreground: bool = False) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        img = img.squeeze().detach().cpu().numpy()
    mask = detect_spots(img)
    if check_foreground:
        foreground = np.count_nonzero(mask)
        pixels = img.shape[0] * img.shape[1]
        ratio = foreground / pixels 
        print(f"Image ratio: {ratio}")
        if ratio < 0.06:
            return None, None, None
    mask_label, num_proteins = measure.label(mask, return_num=True)
    props = measure.regionprops(mask_label, intensity_image=img)
    coordinates = np.array([p.weighted_centroid for p in props])
    features = np.zeros((len(props), 6))
    for i, prop in enumerate(props):
        features[i, 0] = prop.area
        features[i, 1] = prop.perimeter
        features[i, 2] = prop.mean_intensity 
        features[i, 3] = prop.eccentricity  
        features[i, 4] = prop.solidity
        coords = np.array(prop.weighted_centroid)
        distances = cdist(coords.reshape(1, -1), coordinates)
        distances = np.sort(distances)
        if distances.shape[1] == 1:
            nn_dist = 224.0
            features[i, 5] = nn_dist 
        else:
            nn_dist = distances[:, 1] # omitting itself
            features[i, 5] = nn_dist.item()
    
    mean_features = np.mean(features, axis=0)
    if np.any(np.isnan(mean_features)):
        print("\n\n")
        print("NaN features")
        print(features)
        print("\n\n")
        print(mean_features)
        exit()
    mean_features = np.r_[mean_features, num_proteins]
    return props, features, mean_features

def plot_sanity_check(block_features: np.ndarray, mg_features: np.ndarray):
    keys = ["DIV14", "DIV5"]
    os.makedirs(f"./{args.boundary}-experiment/features", exist_ok=True)
    np.savez(f"./{args.boundary}-experiment/features/train-features.npz", block_features=block_features, mg_features=mg_features)
    features = ["area", "perimeter", "mean_intensity", "eccentricity", "solidity", "1nn_dist", "num_proteins"]
    for i, f in enumerate(features):
        data = [ary[:, i] for ary in [block_features, mg_features]]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        parts = ax.violinplot(data, positions=[1.0, 1.6], showmeans=True)
        for i, pc in enumerate(parts['bodies']):
            color = "fuchsia" if i == 0 else "dodgerblue"
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)

        parts['cmeans'].set_color('black')
        parts['cmeans'].set_color('black')
        parts['cbars'].set_color('black')
        parts['cmins'].set_color('black')
        parts['cmaxes'].set_color('black')  
        plt.ylabel(f)
        plt.xticks([1.0, 1.6], ["DIV14", "DIV5"])
        # plt.xlim([0.5, 1.0])
        
        fig.savefig(f"./{args.boundary}-experiment/features/{args.weights}-{f}.pdf", dpi=1200, bbox_inches='tight')
        plt.close(fig)

def plot_distance_distribution(distances_to_boundary: dict):
    key1, key2 = list(distances_to_boundary.keys())
    os.makedirs(f"./{args.boundary}-experiment/distributions", exist_ok=True)
    np.savez(f"./{args.boundary}-experiment/distributions/{args.weights}-{args.boundary}-distance_distribution.npz", **distances_to_boundary)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    m = min(min(distances_to_boundary[key1]), min(distances_to_boundary[key2]))
    M = max(max(distances_to_boundary[key1]), max(distances_to_boundary[key2]))

    ax.hist(distances_to_boundary["DIV14"], bins=np.linspace(m, M, 50), alpha=0.5, color='fuchsia', label="DIV14")
    ax.hist(distances_to_boundary["DIV5"], bins=np.linspace(m, M, 50), alpha=0.5, color='dodgerblue', label="DIV5")
    ax.axvline(0.0, color='black', linestyle='--', label="Decision boundary")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.savefig(f"./{args.boundary}-experiment/distributions/{args.weights}-{args.boundary}-distance_distribution.pdf", dpi=1200, bbox_inches="tight")
    plt.close(fig)

def load_distance_distribution() -> np.ndarray:
    data = np.load(f"./{args.boundary}-experiment/distributions/{args.weights}-{args.boundary}-distance_distribution.npz")

    scores = np.abs(data[args.direction])

    avg, std = np.mean(scores), np.std(scores)
    distance_max = np.max(scores)
    return avg - (3*std), distance_max * 4

def plot_features(features: np.ndarray, distances: np.ndarray, index: int):
    os.makedirs(f"./{args.boundary}-experiment/examples", exist_ok=True)
    
    features_min = features.min(axis=0)
    features_max = features.max(axis=0)
    for i in range(features.shape[1]):
        features[:, i] = (features[:, i] - features_min[i]) / (features_max[i] - features_min[i])

    fig = plt.figure(figsize=(5,5))
    plt.imshow(features, cmap='viridis')
    plt.yticks([0, 1, 2, 3, 4, 5], ["DIV5" if args.direction == "DIV14" else "DIV14", "1", "2", "3", "4", "5"])
    plt.xticks([0, 1, 2, 3, 4, 5, 6], ["area", "perimeter","mean intensity", "eccentricity", "solidity", "1nn_dist", "num_proteins"], rotation=-45)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
    plt.colorbar()
    fig.savefig(f"./{args.boundary}-experiment/examples/{args.weights}-features_{index}_to{args.direction}.pdf", dpi=1200, bbox_inches='tight')

def save_examples(samples, distances, index):
    os.makedirs(f"./{args.boundary}-experiment/examples", exist_ok=True)

    N = len(samples)
    fig, axs = plt.subplots(1, N, figsize=(10, 5))
    for i, (s, d) in enumerate(zip(samples, distances)):
        if s.shape[0] == 3:
            s = s[0, :, :]
        axs[i].imshow(s, cmap='hot', vmin=0.0, vmax=0.8)
        axs[i].set_title("Distance: {:.2f}".format(d))
        axs[i].axis("off")
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
    fig.savefig(f"./{args.boundary}-experiment/examples/{args.weights}-image_{index}_to{args.direction}.pdf", dpi=1200, bbox_inches='tight')
    plt.close(fig)

def save_raw_images(samples, distances, index):
    os.makedirs(f"./{args.boundary}-experiment/examples/raw", exist_ok=True)
    os.makedirs(f"./{args.boundary}-experiment/examples/raw-tif", exist_ok=True)
    
    cmap = plt.get_cmap('hot')
    norm = Normalize(vmin=0.0, vmax=1.0, clip=True)

    for i, (s, d) in enumerate(zip(samples, distances)):
        if s.shape[0] == 3:
            s = s[0, :, :]

        tifffile.imwrite(
            f"./{args.boundary}-experiment/examples/raw-tif/{args.weights}-image_{index}_to{args.direction}_{i}.tif",
            s.astype(np.float32)
        )

        img = Image.fromarray((cmap(norm(s)) * 255).astype(np.uint8))
        img.save(f"./{args.boundary}-experiment/examples/raw/{args.weights}-image_{index}_to{args.direction}_{i}.png")

def plot_results() -> None:
    os.makedirs(f"./{args.boundary}-experiment/features", exist_ok=True)

    features = np.load(f"/home/frederic/flc-dataset/experiments/diffusion-experiments/lerp-results/wavelet_features/MAE_SMALL_STED_activity_all_to{args.direction}_RESULTS.npz")
    num_proteins = np.load(f"/home/frederic/flc-dataset/experiments/diffusion-experiments/lerp-results/wavelet_features/MAE_SMALL_STED_activity_all_to{args.direction}_NUM_PROTEINS.npz")
    feature_names = ["area", "perimeter", "mean_intensity", "eccentricity", "solidity", "1nn_dist"]
    keys = list(features.keys())
    train_features = np.load(f"./{args.boundary}-experiment/features/train-features.npz")
    block_features, mg_features = train_features["block_features"], train_features["mg_features"]
    # block_features, mg_features = np.array(block_features), np.array(mg_features)   
    for i, f in enumerate(feature_names):
        data = [features[k][:, i] for k in keys] 
        block_data = block_features[:, i]
        mg_data = mg_features[:, i] 
        data.insert(0, block_data)
        data.append( mg_data)
        
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        parts = ax.boxplot(data, medianprops={'color': 'black'}, showfliers=False, patch_artist=True)
        N = len(data)
        for i, pc in enumerate(parts['boxes']):
            color = "grey"
            if i == 0:
                color = "fuchsia" 
            if i == N - 1:
                color = "dodgerblue"
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8], ["Block", "0", "1", "2", "3", "4", "5", "0MgGlyBic"])
        fig.savefig(f"./{args.boundary}-experiment/results/{args.weights}-{f}-with-train.pdf", dpi=1200, bbox_inches='tight')
        plt.close(fig)

    data = [num_proteins[k] for k in keys]
    block_data = block_features[:, -1]
    mg_data = mg_features[:, -1]
    data.insert(0, block_data)
    data.append(mg_data) 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    parts = ax.boxplot(data, medianprops={'color': 'black'}, showfliers=False, patch_artist=True)
    for i, pc in enumerate(parts['boxes']):
        color = "grey"
        if i == 0:
            color = "fuchsia" 
        if i == N - 1:
            color = "dodgerblue"
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8], ["Block", "0", "1", "2", "3", "4", "5", "0MgGlyBic"])
    fig.savefig(f"./{args.boundary}-experiment/results/num_proteins-with-train.pdf", dpi=1200, bbox_inches='tight')
    plt.close(fig)


def main():
    set_seeds(args.seed)
    if args.figure:
        plot_results()
    elif args.sanity_check:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        boundary, intercept, norm = load_boundary() 
        latent_encoder, model_config = get_pretrained_model_v2(
            name=args.latent_encoder,
            weights=args.weights,
            path=None,
            mask_ratio=0.0,
            pretrained=True if "imagenet" in args.weights.lower() else False,
            in_channels=3 if "imagenet" in args.weights.lower() else 1,
            as_classifier=True,
            blocks="all",
            num_classes=4
        )
        latent_encoder.to(DEVICE)
        latent_encoder.eval()
        dataset = ALSDataset(
            tarpath="/home-local/Frederic/Datasets/ALS/catalog/PLKO_train.tar",
        ) 
        N = len(dataset)
        indices = np.arange(N)
        np.random.shuffle(indices)
        counters = {"DIV14": 0, "DIV5": 0}
        distances_to_boundary = {"DIV14": [], "DIV5": []}
        features = {"DIV14" : None, "DIV5": None}
        with torch.no_grad():
            for i, idx in tqdm(enumerate(indices), total=N):
                img, metadata = dataset[idx]
                img = img.to(DEVICE)
                label = metadata["label"]

                original = img.squeeze().detach().cpu().numpy()
                rprops, _, mean_features = extract_features(original, check_foreground=True)

                if mean_features is None:
                    print("Not enough foreground, skipping...")
                    continue 

                torch_img = img.clone().detach().unsqueeze(0).to(DEVICE)
                latent_code = latent_encoder.forward_features(torch_img)
                numpy_code = latent_code.detach().cpu().numpy()

                d = numpy_code.dot(boundary.T) + intercept
                d = d[0][0] 

                mean_features = mean_features.reshape(1, -1) 
                if counters[label] == 0:
                    features[label] = mean_features
                else:
                    features[label] = np.r_[features[label], mean_features]
                distances_to_boundary[label].append(d)
                counters[label] += 1
            
        for key, values in features.items():
            print(f"Key: {key}, Features: {values.shape}")
        plot_sanity_check(features["DIV14"], features["DIV5"])
        plot_distance_distribution(distances_to_boundary)

    else:
        RESULTS = {} 
        NUM_PROTEINS = {
            "original": [],
            "lerp_1": [],
            "lerp_2": [],
            "lerp_3": [],
            "lerp_4": [],
            "lerp_5": [],
        }

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        boundary, intercept, norm = load_boundary()
        distance_min, distance_max = load_distance_distribution()
        print(f"--- Moving from 0.0 to {distance_max} ---")
        latent_encoder, model_config = get_pretrained_model_v2(
            name=args.latent_encoder,
            weights=args.weights,
            path=None,
            mask_ratio=0.0,
            pretrained=True if "imagenet" in args.weights.lower() else False,
            in_channels=3 if "imagenet" in args.weights.lower() else 1,
            as_classifier=True,
            blocks="all",
            num_classes=4
        )
        denoising_model = UNet(
            dim=64,
            channels=1,
            dim_mults=(1,2,4),
            cond_dim=model_config.dim,
            condition_type="latent",
            num_classes=4
        )
        diffusion_model = DDPM(
            denoising_model=denoising_model,
            timesteps=args.timesteps,
            beta_schedule="linear",
            condition_type="latent",
            latent_encoder=latent_encoder,
        )

        ckpt = torch.load(f"{args.ckpt_path}/{args.weights}/checkpoint-69.pth")
        diffusion_model.load_state_dict(ckpt["state_dict"])
        diffusion_model.to(DEVICE)
        dataset = ALSDataset(
            tarpath="/home-local/Frederic/Datasets/ALS/catalog/PLKO_test.tar",
        )
        N = len(dataset)
        indices = np.arange(N)
        np.random.shuffle(indices)
        counter = 0

        with open(f"./{args.boundary}-experiment/embeddings/{args.weights}-{args.boundary}-labels_train.json", "r") as f:
            target_labels = json.load(f)

        print(target_labels)

        for i in tqdm(indices):
            rprops = [] 
            distances = []
            features = []
            all_features = np.zeros((args.n_steps+1, 7))
            # if counter >= args.num_samples:
            #     break 
            img, metadata = dataset[i]
            label = metadata["label"]
            current_label = target_labels[label]
            print(label, current_label)

            multiplier = 1 if current_label == 0 else -1
            if args.boundary == "als" and args.direction == label: 
                print(f"Skipping {i} because condition is {label} and target is {args.direction}")
                continue 

            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            original = img.squeeze().detach().cpu().numpy()
            original_rprops, original_features, original_mean_features = extract_features(original, check_foreground=True)

            if original_features is None:
                print("Not enough foreground, skipping...")
                continue 

            RESULTS["original"] = original_features if counter == 0 else np.r_[RESULTS["original"], original_features]
            NUM_PROTEINS["original"].append(original_mean_features[6])

            all_features[0] = original_mean_features 
            features.append(original_features)

            seed_offset = hash(args.direction) % (2**32-1)
            set_seeds(args.seed + i + seed_offset)                        

            latent_code = diffusion_model.latent_encoder.forward_features(img)
            numpy_code = latent_code.detach().cpu().numpy() 

            samples = [original]
            distances.append(0.0)
            rprops.append(original_rprops)

            lerped_codes, d = linear_interpolate(latent_code=numpy_code, boundary=boundary, intercept=intercept,start_distance=multiplier*0.0, end_distance=multiplier*distance_max, steps=args.n_steps)
            print(d)

            for c, code in enumerate(lerped_codes):
                lerped_code = torch.tensor(code, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                lerped_sample = diffusion_model.p_sample_loop(shape=(img.shape[0], 1, img.shape[2], img.shape[3]), cond=lerped_code, progress=True)
                lerped_sample_numpy = lerped_sample.squeeze().detach().cpu().numpy()
                lerped_rprops, lerped_sample_features, lerped_sample_mean_features = extract_features(lerped_sample_numpy)
                RESULTS["lerp_" + str(c+1)] = lerped_sample_features if counter == 0 else np.r_[RESULTS["lerp_" + str(c+1)], lerped_sample_features]
                samples.append(lerped_sample_numpy)
                NUM_PROTEINS["lerp_" + str(c+1)].append(lerped_sample_mean_features[6])
                all_features[c+1] = lerped_sample_mean_features
                features.append(lerped_sample_features)
                distances.append(abs(d[c][0]))
                rprops.append(lerped_rprops)
         

            distances = np.array(distances)
            counter += 1

            plot_features(features=all_features, distances=distances, index=counter)
            save_examples(samples, distances, counter)
            save_raw_images(samples, distances, counter)

        os.makedirs(f"./{args.boundary}-experiment/results", exist_ok=True)
        np.savez(f"./{args.boundary}-experiment/results/{args.weights}_{args.boundary}_all_to{args.direction}_RESULTS.npz", **RESULTS)
        np.savez(f"./{args.boundary}-experiment/results/{args.weights}_{args.boundary}_all_to{args.direction}_NUM_PROTEINS.npz", **NUM_PROTEINS)
                
        

if __name__ == "__main__":
    main()
                
                
        

    


