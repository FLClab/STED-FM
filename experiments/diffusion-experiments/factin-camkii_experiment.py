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
from attribute_datasets import ProteinActivityDataset
import sys 
from scipy.spatial.distance import cdist
from skimage import filters
import glob 
import pickle
from PIL import Image
import tifffile

sys.path.insert(0, "../")
from DEFAULTS import BASE_PATH, COLORS 
from model_builder import get_pretrained_model_v2 
from utils import set_seeds
from datasets import FactinCaMKIIDataset

CONDITIONA = "CTRL"
CONDITIONB = "shRNA"

parser = argparse.ArgumentParser()
parser.add_argument("--latent-encoder", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--timesteps", type=int, default=1000)
parser.add_argument("--boundary", type=str, default="factin-camkii")
parser.add_argument("--num-samples", type=int, default=20)
parser.add_argument("--ckpt-path", type=str, default=f"{BASE_PATH}/baselines/DiffusionModels/latent-guidance")
parser.add_argument("--figure", action="store_true")
parser.add_argument("--sanity-check", action="store_true")
parser.add_argument("--direction", type=str, default="shRNA")
parser.add_argument("--n-steps", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()


def compute_confidence_intervals(all_scores: np.ndarray, confidence: float = 0.80) -> tuple:
    """Compute confidence intervals for scores at each step.
    
    Args:
        all_scores: Array of shape (num_samples, num_steps) containing scores
        confidence: Confidence level (default: 0.95 for 95% CI)
    
    Returns:
        tuple: (lower_bounds, upper_bounds) arrays for the confidence intervals
    """
    from scipy import stats
    
    # Calculate mean and standard error for each step
    means = np.mean(all_scores, axis=0)
    se = stats.sem(all_scores, axis=0)
    
    # Calculate confidence intervals
    ci = stats.t.interval(confidence, len(all_scores)-1, loc=means, scale=se)
    
    return ci[0], ci[1] 


def linear_interpolate(latent_code,
                       boundary,
                       intercept,
                       norm=1.0,
                       start_distance=-4.0,
                       end_distance=4.0,
                       steps=8):
  assert (latent_code.shape[0] == 1 and boundary.shape[0] == 1 and
          len(boundary.shape) == 2 and
          boundary.shape[1] == latent_code.shape[-1])

  image_distance = latent_code.dot(boundary.T) + intercept
  end_distance = end_distance - image_distance
  if steps < 2:
      linspace = np.array([end_distance])
  else:
    linspace = np.linspace(start_distance, end_distance, steps)[1:]

  if len(latent_code.shape) == 2:
    linspace = linspace.reshape(-1, 1).astype(np.float32)
    return latent_code + linspace * boundary * norm, linspace, image_distance[0][0]
  if len(latent_code.shape) == 3:
    linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
    return latent_code + linspace * boundary.reshape(1, 1, -1) * norm, linspace, image_distance[0][0]
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

    if check_foreground:
        mask = img > filters.threshold_otsu(img)
        foreground = np.count_nonzero(mask)
        pixels = img.shape[0] * img.shape[1]
        ratio = foreground / pixels 
        if ratio < 0.1:
            return False

    return True


def plot_features(features: np.ndarray, distances: np.ndarray, index: int):
    os.makedirs(f"./{args.boundary}-experiment/examples", exist_ok=True)
    
    features_min = features.min(axis=0)
    features_max = features.max(axis=0)
    for i in range(features.shape[1]):
        features[:, i] = (features[:, i] - features_min[i]) / (features_max[i] - features_min[i])

    fig = plt.figure(figsize=(5,5))
    plt.imshow(features, cmap='viridis')
    plt.yticks([0, 1, 2, 3, 4, 5], [CONDITIONA if args.direction == CONDITIONB else CONDITIONB, "1", "2", "3", "4", "5"])
    plt.xticks([0, 1, 2, 3, 4, 5, 6], ["area", "perimeter","mean intensity", "eccentricity", "solidity", "1nn_dist", "num_proteins"], rotation=-45)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
    plt.colorbar()
    fig.savefig(f"./{args.boundary}-experiment/examples/{args.weights}-features_{index}_to{args.direction}.pdf", dpi=1200, bbox_inches='tight')
    # fig.savefig("./temp.pdf", dpi=1200)

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

def save_examples(samples, distances, index):
    os.makedirs(f"./{args.boundary}-experiment/examples", exist_ok=True)

    N = len(samples)
    fig, axs = plt.subplots(1, N, figsize=(10, 5))
    for i, (s, d) in enumerate(zip(samples, distances)):
        if s.shape[0] == 3:
            s = s[0, :, :]
        axs[i].imshow(s, cmap='hot', vmin=0.0, vmax=1.0)
        axs[i].set_title("Distance: {:.2f}".format(d))
        axs[i].axis("off")
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
    fig.savefig(f"./{args.boundary}-experiment/examples/{args.weights}-image_{index}_to{args.direction}.pdf", dpi=1200, bbox_inches='tight')
    plt.close(fig)

def plot_distance_distribution(distances_to_boundary: dict):
    key1, key2 = list(distances_to_boundary.keys())
    os.makedirs(f"./{args.boundary}-experiment/distributions", exist_ok=True)
    np.savez(f"./{args.boundary}-experiment/distributions/{args.weights}-{args.boundary}-distance_distribution.npz", **distances_to_boundary)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    m = min(min(distances_to_boundary[key1]), min(distances_to_boundary[key2]))
    M = max(max(distances_to_boundary[key1]), max(distances_to_boundary[key2]))

    ax.hist(distances_to_boundary[CONDITIONA], bins=np.linspace(m, M, 50), alpha=0.5, color='fuchsia', label=CONDITIONA)
    ax.hist(distances_to_boundary[CONDITIONB], bins=np.linspace(m, M, 50), alpha=0.5, color='dodgerblue', label=CONDITIONB)
    ax.axvline(0.0, color='black', linestyle='--', label="Decision boundary")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.savefig(f"./{args.boundary}-experiment/distributions/{args.weights}-{args.boundary}-distance_distribution.pdf", dpi=1200, bbox_inches="tight")
    plt.close(fig)

def plot_results_old():
    results = np.load(f"/home/frederic/flc-dataset/experiments/diffusion-experiments/lerp-results/wavelet_features/MAE_SMALL_STED_activity_all_to{args.direction}_RESULTS.npz")
    num_proteins = np.load(f"/home/frederic/flc-dataset/experiments/diffusion-experiments/lerp-results/wavelet_features/MAE_SMALL_STED_activity_all_to{args.direction}_NUM_PROTEINS.npz")
    features = ["area", "perimeter", "mean_intensity", "eccentricity", "solidity", "1nn_dist"]
    keys = list(results.keys()) 
    for i, f in enumerate(features):
        ### Protein-wise feature violin plots
        data = [results[k][:, i] for k in keys]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        parts = ax.boxplot(data, medianprops={'color': 'black'}, patch_artist=True)
        
        for pc in parts['boxes']:
            pc.set_facecolor('grey')
            pc.set_alpha(0.7)
    
        ax.set_ylabel(f)
        ax.set_xlabel("Distance")
        ax.set_xticks([1, 2, 3, 4, 5, 6], ["0", "0", "1", "2", "3", "4"])
        
        os.makedirs(f"./{args.boundary}-experiment/features", exist_ok=True)
        fig.savefig(f"./{args.boundary}-experiment/features/{args.weights}-{f}_to{args.direction}.pdf", dpi=1200, bbox_inches='tight')
        plt.close(fig)

        # Image-wise number of proteins violin plot
        data = [num_proteins[k] for k in keys]
        fig = plt.figure()
        ax = fig.add_subplot(111)   
        parts = ax.boxplot(data, medianprops={'color': 'black'}, patch_artist=True)
        
        for pc in parts['boxes']:
            pc.set_facecolor('grey')
            pc.set_alpha(0.7)
        ax.set_ylabel("Number of proteins")
        ax.set_xlabel("Distance")
        ax.set_xticks([1, 2, 3, 4, 5, 6], ["0", "0", "1", "2", "3", "4"])
        
        fig.savefig(f"./{args.boundary}-experiment/features/num_proteins_to{args.direction}.pdf", dpi=1200, bbox_inches='tight')
        plt.close(fig)

def plot_results() -> None:
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
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8], [CONDITIONA, "0", "1", "2", "3", "4", "5", CONDITIONB])
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
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8], [CONDITIONA, "0", "1", "2", "3", "4", "5", CONDITIONB])
    fig.savefig(f"./{args.boundary}-experiment/results/num_proteins-with-train.pdf", dpi=1200, bbox_inches='tight')
    plt.close(fig)

def plot_sanity_check(block_features: np.ndarray, mg_features: np.ndarray):
    keys = [CONDITIONA, CONDITIONB]
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
        plt.xticks([1.0, 1.6], [CONDITIONA, CONDITIONB])
        # plt.xlim([0.5, 1.0])
        
        fig.savefig(f"./{args.boundary}-experiment/features/{args.weights}-{f}.pdf", dpi=1200, bbox_inches='tight')
        plt.close(fig)

def plot_feature_path():
    # TODO: update now that we don't include the sample features 
    mg_path = np.load(f"./{args.boundary}-experiment/results/{args.weights}_{args.boundary}_all_toGluGly_RESULTS.npz")
    block_path = np.load(f"./{args.boundary}-experiment/results/{args.weights}_{args.boundary}_all_toBlock_RESULTS.npz")
    mg_proteins = np.load(f"./{args.boundary}-experiment/results/{args.weights}_{args.boundary}_all_toGluGly_NUM_PROTEINS.npz")
    block_proteins = np.load(f"./{args.boundary}-experiment/results/{args.weights}_{args.boundary}_all_toBlock_NUM_PROTEINS.npz")
    keys_to = list(mg_path.keys())
    keys_back = list(block_path.keys())
    keys_to.remove("sample")
    keys_to.remove("original")
    keys_back.remove("original")
    keys_back.remove("sample")

    train_features = np.load(f"/home/frederic/flc-dataset/experiments/diffusion-experiments/lerp-results/wavelet_features/sanity-check/train-features.npz")
    block_features = train_features["block_features"]
    mg_features = train_features["mg_features"]
    source_area = block_features[:, 0]
    source_proteins = block_features[:, -1]
    destination_area = mg_features[:, 0]
    destination_proteins = mg_features[:, -1]

    area_to = [mg_path[k][:, 0] for k in keys_to] 
    area_back = [block_path[k][:, 0] for k in keys_back]
    proteins_to = [mg_proteins[k] for k in keys_to]
    proteins_back = [block_proteins[k] for k in keys_back]

    area_data = [source_area] + area_to + [destination_area] + area_back + [source_area]
    protein_data = [source_proteins] + proteins_to + [destination_proteins] + proteins_back + [source_proteins]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    parts = ax.boxplot(area_data, medianprops={'color': 'black'}, patch_artist=True)
    for pc in parts['boxes']:
        pc.set_facecolor('grey')
        pc.set_alpha(0.7)
        
    ax.set_ylabel("Area")
    ax.set_xlabel("Distance")
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [CONDITIONA, "1", "2", "3", "4", CONDITIONB, "4", "3", "2", "1", CONDITIONA])
    
    fig.savefig(f"./{args.boundary}-experiment/features/area_full_path.pdf", dpi=1200, bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    parts = ax.boxplot(protein_data, medianprops={'color': 'black'}, patch_artist=True)
    for pc in parts['boxes']:
        pc.set_facecolor('grey')
        pc.set_alpha(0.7)
        
    ax.set_ylabel("# Proteins")
    ax.set_xlabel("Distance")
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [CONDITIONA, "1", "2", "3", "4", CONDITIONB, "4", "3", "2", "1", CONDITIONA])
    
    fig.savefig(f"./{args.boundary}-experiment/features/num_proteins_full_path.pdf", dpi=1200, bbox_inches='tight')
    plt.close(fig)

def load_distance_distribution() -> np.ndarray:
    data = np.load(f"./{args.boundary}-experiment/distributions/{args.weights}-{args.boundary}-distance_distribution.npz")

    scores = np.abs(data[args.direction])

    avg, std = np.mean(scores), np.std(scores)
    distance_max = np.max(scores)
    distance_min = np.min(scores)
    return distance_min, distance_max

def main():


    set_seeds(args.seed)

    if args.figure:
        plot_results()
        # plot_feature_path()
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
        dataset = FactinCaMKIIDataset(
            os.path.join(BASE_PATH, "evaluation-data", "factin-camkii", f"train-dataset.tar"),
            transform=None,
            n_channels=3 if "imagenet" in args.weights.lower() else 1,
        )
        N = len(dataset)
        indices = np.arange(N)
        np.random.shuffle(indices)

        counters = {CONDITIONA : 0, CONDITIONB: 0}
        distances_to_boundary = {CONDITIONA: [], CONDITIONB: []}
        features = {CONDITIONA : None, CONDITIONB: None}
        with torch.no_grad():
            for i, idx in tqdm(enumerate(indices), total=N):
                
                img, metadata = dataset[idx]
                img = img.to(DEVICE)
                label = metadata["condition"]
            

                original = img.squeeze().detach().cpu().numpy() 

                torch_img = img.clone().detach().unsqueeze(0).to(DEVICE)
                latent_code = latent_encoder.forward_features(torch_img)
                numpy_code = latent_code.detach().cpu().numpy()

                d = numpy_code.dot(boundary.T) + intercept
                d = d[0][0]
            
                distances_to_boundary[label].append(d)
                counters[label] += 1
        
        for key, values in distances_to_boundary.items():
            distances_to_boundary[key] = np.array(values)
            print(f"--- {key} ---")
            print(f"Mean: {np.mean(values)}")
            print(f"Std: {np.std(values)}")
            print(f"Max: {np.max(values)}")
            print(f"Min: {np.min(values)}")

        print("Plotting...")
        plot_distance_distribution(distances_to_boundary)

    else:

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        boundary, intercept, norm = load_boundary()
        distance_min, distance_max = load_distance_distribution()
        print(f"--- Moving from 0.0 to {distance_max} ---")
        print(f"--- Norm of boundary: {norm} ---")
        print(f"--- Intercept: {intercept} ---")

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
        dataset = FactinCaMKIIDataset(
            os.path.join(BASE_PATH, "evaluation-data", "factin-camkii", f"test-dataset.tar"),
            transform=None,
            n_channels=3 if "imagenet" in args.weights.lower() else 1,
        )
        N = len(dataset) 
        indices = np.arange(N)
        np.random.shuffle(indices)
        counter = 0

        with open(f"./{args.boundary}-experiment/embeddings/{args.weights}-{args.boundary}-labels_train.json", "r") as f:
            target_labels = json.load(f)

        print(target_labels)

        with torch.no_grad():
            for i in tqdm(indices):
                rprops = []
                distances = [] 
                features = []
                all_features = np.zeros((args.n_steps+1, 7))
                if counter >= args.num_samples:
                    break 
                img, metadata = dataset[i]
                label = metadata["label"]


                condition = metadata["condition"]
                target_label = target_labels[condition]

                multiplier = 1 if target_label == 0 else -1
                if args.direction == condition:
                    print(f"Skipping {i} because condition is {condition} and direction is {args.direction}")
                    continue 

                if "imagenet" in args.weights.lower():
                    img = torch.tensor(img.clone().detach(), dtype=torch.float32).repeat(3, 1, 1).unsqueeze(0).to(DEVICE) 
                else:
                    img = torch.tensor(img.clone().detach(), dtype=torch.float32).unsqueeze(0).to(DEVICE)

                original = img.squeeze().detach().cpu().numpy()
                features = extract_features(original, check_foreground=True)
                
                if not features:
                    print("Not enough foreground, skipping...")
                    continue
            
                # Ensures reproducibility
                seed_offset = hash(args.direction) % (2**32-1)
                set_seeds(args.seed + i + seed_offset)

                latent_code = diffusion_model.latent_encoder.forward_features(img)
                numpy_code = latent_code.detach().cpu().numpy()

                samples = [original]


                lerped_codes, d, image_distance = linear_interpolate(latent_code=numpy_code, boundary=boundary, intercept=intercept, norm=norm, start_distance=multiplier * 0.0, end_distance=multiplier * distance_max, steps=args.n_steps+1)
                distances.append(image_distance)
                print(d)

                # lerped_samples = diffusion_model.p_sample_loop(
                #     shape=(len(lerped_codes), 1, img.shape[2], img.shape[3]),
                #     cond=torch.tensor(lerped_codes, dtype=torch.float32).to(DEVICE),
                #     progress=True
                # )
                # lerped_samples_numpy = lerped_samples.squeeze().detach().cpu().numpy()
                # samples.extend(lerped_samples_numpy)
                # distances.extend(d.ravel())

                for c, code in enumerate(lerped_codes):
                    lerped_code = torch.tensor(code, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    lerped_sample = diffusion_model.p_sample_loop(shape=(img.shape[0], 1, img.shape[2], img.shape[3]), cond=lerped_code, progress=True)
                    lerped_sample_numpy = lerped_sample.squeeze().detach().cpu().numpy()

                    samples.append(lerped_sample_numpy)
                    distances.append(image_distance + d[c][0])
            
                distances = np.array(distances)
                counter += 1

                # plot_features(features=all_features, distances=distances, index=counter)
                save_examples(samples, distances, counter)
                save_raw_images(samples, distances, counter)

            # os.makedirs(f"./{args.boundary}-experiment/results", exist_ok=True)
            # np.savez(f"./{args.boundary}-experiment/results/{args.weights}_{args.boundary}_all_to{args.direction}_RESULTS.npz", **RESULTS)
            # np.savez(f"./{args.boundary}-experiment/results/{args.weights}_{args.boundary}_all_to{args.direction}_NUM_PROTEINS.npz", **NUM_PROTEINS)


if __name__ == "__main__":
    main()

