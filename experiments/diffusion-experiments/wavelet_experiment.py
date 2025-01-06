import numpy as np 
import matplotlib.pyplot as plt 
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
import glob 
sys.path.insert(0, "../")
from DEFAULTS import BASE_PATH, COLORS 
from model_builder import get_pretrained_model_v2 

parser = argparse.ArgumentParser()
parser.add_argument("--latent-encoder", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--timesteps", type=int, default=1000)
parser.add_argument("--boundary", type=str, default="activity")
parser.add_argument("--num-samples", type=int, default=5)
parser.add_argument("--ckpt-path", type=str, default="/home-local/Frederic/baselines/DiffusionModels/latent-guidance")
parser.add_argument("--figure", action="store_true")
parser.add_argument("--direction", type=str, default="0Mg")
args = parser.parse_args()

def linear_interpolate(latent_code,
                       boundary,
                       start_distance=-4.0,
                       end_distance=4.0,
                       steps=8):
  assert (latent_code.shape[0] == 1 and boundary.shape[0] == 1 and
          len(boundary.shape) == 2 and
          boundary.shape[1] == latent_code.shape[-1])

  linspace = np.linspace(start_distance, end_distance, steps)
  if len(latent_code.shape) == 2:
    linspace = linspace - latent_code.dot(boundary.T)
    linspace = linspace.reshape(-1, 1).astype(np.float32)
    return latent_code + linspace * boundary, linspace
  if len(latent_code.shape) == 3:
    linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
    return latent_code + linspace * boundary.reshape(1, 1, -1), linspace
  raise ValueError(f'Input `latent_code` should be with shape '
                   f'[1, latent_space_dim] but {latent_code.shape} was received.')

def load_boundary(boundary: str) -> np.ndarray:
    print(f"--- Loading boundary trained from {args.weights} embeddings ---")
    return np.load(f"./lerp-results/boundaries/{args.boundary}/{args.weights}_{args.boundary}_boundary.npz")["boundary"]

def extract_features(img: Union[np.ndarray, torch.Tensor], check_foreground: bool = False) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        img = img.squeeze().detach().cpu().numpy()
    mask = detect_spots(img)
    if check_foreground:
        foreground = np.count_nonzero(mask)
        pixels = img.shape[0] * img.shape[1]
        ratio = foreground / pixels 
        if ratio < 0.05:
            return None, None
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
        nn_dist = distances[:, 1] # omitting itself
        features[i, 5] = nn_dist
    
    mean_features = np.mean(features, axis=0)
    if np.any(np.isnan(mean_features)):
        print("NaN features")
        exit()
    mean_features = np.r_[mean_features, num_proteins]
    return features, mean_features


def plot_features(features: np.ndarray, distances: np.ndarray, index: int):
    features_min = features.min(axis=0)
    features_max = features.max(axis=0)
    for i in range(features.shape[1]):
        features[:, i] = (features[:, i] - features_min[i]) / (features_max[i] - features_min[i])

    fig = plt.figure(figsize=(5,5))
    plt.imshow(features, cmap='viridis')
    plt.yticks([0, 1, 2, 3, 4, 5], ["0", "0", "1", "2", "3", "4"])
    plt.xticks([0, 1, 2, 3, 4, 5, 6], ["area", "perimeter","mean intensity", "eccentricity", "solidity", "1nn_dist", "num_proteins"], rotation=-45)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
    fig.savefig(f"./lerp-results/wavelet_features/{args.weights}_{args.boundary}_wavelet_{index}_to{args.direction}.png", bbox_inches='tight')

def save_examples(samples, distances, index):
    N = len(samples)
    fig, axs = plt.subplots(1, N, figsize=(10, 5))
    for i, (s, d) in enumerate(zip(samples, distances)):
        if s.shape[0] == 3:
            s = s[0, :, :]
        axs[i].imshow(s, cmap='hot', vmin=0.0, vmax=1.0)
        axs[i].set_title("Distance: {:.2f}".format(d))
        axs[i].axis("off")
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
    fig.savefig(f"./lerp-results/examples/activity/{args.weights}-image_{index}_to{args.direction}.pdf", dpi=1200, bbox_inches='tight')
    plt.close(fig)


def plot_results():
    # TODO
    pass 

def main():
    if args.figure:
        plot_results()
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        boundary = load_boundary(args.boundary)
        latent_encoder, model_config = get_pretrained_model_v2(
            name=args.latent_encoder,
            weights=args.weights,
            path=None, 
            mask_ratio=0.0,
            pretrained=False,
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
        dataset = ProteinActivityDataset(
            h5file=f"/home-local/Frederic/Datasets/evaluation-data/NeuralActivityStates/NAS_test.hdf5",
            num_samples=None,
            transform=None,
            n_channels=3 if "imagenet" in args.weights.lower() else 1,
            num_classes=2,
            protein_id=3,
            balance=True,
            keepclasses=[0, 1]
        )
        N = len(dataset) 
        indices = np.arange(N)
        np.random.shuffle(indices)
        counter = 0
        n_steps = 4  
        # all_features = np.zeros((args.num_samples, n_steps+2, 7))
        all_distances = np.zeros((args.num_samples, n_steps+2))

        for i in tqdm(indices):
            distances = [] 
            features = []
            all_features = np.zeros((n_steps+2, 7))
            if counter >= args.num_samples:
                break 
            img, metadata = dataset[i]
            label = metadata["label"]
            target_label = 0 if args.direction == "0Mg" else 1
            multiplier = 1 if args.direction == "0Mg" else -1
            if args.boundary == "activity" and label != target_label:
                continue 

            if "imagenet" in args.weights.lower():
                img = torch.tensor(img, dtype=torch.float32).repeat(3, 1, 1).unsqueeze(0).to(DEVICE) 
            else:
                img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            original = img.squeeze().detach().cpu().numpy()
            original_features, original_mean_features = extract_features(original, check_foreground=True)

            if original_features is None:
                print("Not enough foreground, skipping...")
                continue
            all_features[0] = original_mean_features
            features.append(original_features)

            latent_code = diffusion_model.latent_encoder.forward_features(img)
            numpy_code = latent_code.detach().cpu().numpy()
            original_sample = diffusion_model.p_sample_loop(shape=(img.shape[0], 1, img.shape[2], img.shape[3]), cond=latent_code, progress=True)

            original_sample_numpy = original_sample.squeeze().detach().cpu().numpy() 
            samples = [original, original_sample_numpy]
            sample_features, sample_mean_features = extract_features(original_sample_numpy)
            all_features[1] = sample_mean_features
            features.append(sample_features)

            distances.extend([0.0, 0.0]) 

            lerped_codes, d = linear_interpolate(latent_code=numpy_code, boundary=boundary, start_distance=multiplier*1.0, end_distance=multiplier*4.0, steps=n_steps)

            for c, code in enumerate(lerped_codes):
                lerped_code = torch.tensor(code, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                lerped_sample = diffusion_model.p_sample_loop(shape=(img.shape[0], 1, img.shape[2], img.shape[3]), cond=lerped_code, progress=True)
                lerped_sample_numpy = lerped_sample.squeeze().detach().cpu().numpy()
                lerped_sample_features, lerped_sample_mean_features = extract_features(lerped_sample_numpy)
                samples.append(lerped_sample_numpy)
                all_features[c+2] = lerped_sample_mean_features
                features.append(lerped_sample_features)
                distances.append(abs(d[c][0]))
         

            distances = np.array(distances)

            all_distances[counter] = distances
            counter += 1

            plot_features(features=all_features, distances=distances, index=counter)
            save_examples(samples, distances, counter)
            
            


if __name__ == "__main__":
    main()

