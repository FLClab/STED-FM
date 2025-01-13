import numpy as np
import matplotlib.pyplot as plt 
import argparse 
import torch 
from torch import nn 
from diffusion_models.diffusion.ddpm_lightning import DDPM 
from diffusion_models.diffusion.denoising.unet import UNet 
from tqdm import tqdm 
from attribute_datasets import LowHighResolutionDataset
import sys
from banditopt.objectives import Resolution 
import glob
sys.path.insert(0, "../")
from DEFAULTS import BASE_PATH, COLORS
from model_builder import get_pretrained_model_v2

parser = argparse.ArgumentParser()
parser.add_argument("--latent-encoder", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--timesteps", type=int, default=1000)
parser.add_argument("--boundary", type=str, default="resolution")
parser.add_argument("--num-samples", type=int, default=10)
parser.add_argument("--ckpt-path", type=str, default="/home-local/Frederic/baselines/DiffusionModels/latent-guidance")
parser.add_argument("--figure", action="store_true")
parser.add_argument("--sanity-check", action="store_true")
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

def load_boundary() -> np.ndarray:
    print(f"--- Loading boundary trained from {args.weights} embeddings ---")
    data = np.load(f"./lerp-results/boundaries/{args.boundary}/{args.weights}_{args.boundary}_boundary.npz")
    boundary, intercept, norm = data["boundary"], data["intercept"], data["norm"]
    return boundary, intercept, norm

def compute_resolution(img: np.ndarray) -> float:
    if img.shape[0] == 3:
        img = img[0]
    resolution_objective = Resolution(pixelsize=20e-9)
    resolution = resolution_objective.evaluate([img], None, None, None, None)
    return resolution

def save_examples(samples, distances, resolutions, index):
   N = len(samples)
   fig, axs = plt.subplots(1, N, figsize=(10,5))
   for i, (s, d, r) in enumerate(zip(samples, distances, resolutions)):
       if s.shape[0] == 3:
           s = s[0]
       axs[i].imshow(s, cmap="hot", vmin=0.0, vmax=1.0)
       axs[i].set_title(f"Distance: {d:.2f}\nRes: {r:.2f}")
       axs[i].axis("off")
   plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
   fig.savefig(f"./lerp-results/examples/resolution/{args.weights}-image_{index}.pdf", dpi=1200, bbox_inches="tight")
   plt.close(fig)

def plot_distance_distribution(distances_to_boundary: dict):
    key1, key2 = list(distances_to_boundary.keys())
    np.savez(f"./lerp-results/examples/resolution/sanity-check/{args.weights}-resolution-distance_distribution.npz", key1=distances_to_boundary[key1], key2=distances_to_boundary[key2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(distances_to_boundary["low"], bins=50, alpha=0.5, color='fuchsia', label="Low")
    ax.hist(distances_to_boundary["high"], bins=50, alpha=0.5, color='dodgerblue', label="High")
    ax.axvline(0.0, color='black', linestyle='--', label="Decision boundary")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.savefig(f"./lerp-results/examples/resolution/sanity-check/{args.weights}-resolution-distance_distribution.pdf", dpi=1200, bbox_inches="tight")
    plt.close(fig)

def plot_correlation(all_resolutions, all_distances):
    resolutions = np.mean(all_resolutions, axis=0)
    distances = np.mean(all_distances, axis=0)
    distances = [round(d, 3) for d in distances]
    err = np.std(all_resolutions, axis=0)
    np.savez(f"./lerp-results/correlation/{args.weights}-resolution-correlation.npz", resolutions=resolutions, distances=distances, err=err)
    fig = plt.figure(figsize=(5,5))
    plt.plot(distances, resolutions, c="black")
    plt.fill_between(distances, resolutions-err, resolutions+err, color="black", alpha=0.2)
    plt.xlabel("Distance")
    plt.ylabel("Resolution")
    plt.title("Distance vs Resolution")
    fig.savefig(f"./lerp-results/correlation/{args.weights}-resolution-correlation.png", bbox_inches="tight", dpi=1200)
    plt.close()

def plot_results():
    files = glob.glob(f"./lerp-results/correlation/**-resolution-correlation.npz")
    fig = plt.figure()
    for f in files:
        weight = f.split("/")[-1].split("-")[0].replace("MAE_SMALL_", "")
        data = np.load(f)
        resolutions, distances, err = data["resolutions"], data["distances"], data["err"]
        x = np.arange(distances.shape[0])
        plt.plot(x, resolutions, c=COLORS[weight], label=weight)
    plt.legend()
    plt.xlabel("Distance from original embedding")
    plt.ylabel("Resolution")
    fig.savefig(f"./lerp-results/correlation/resolution/resolution-correlation.pdf", bbox_inches="tight", dpi=1200)
    plt.close()

def load_distance_distribution() -> np.ndarray:
    data = np.load(f"./lerp-results/examples/resolution/sanity-check/{args.weights}-resolution-distance_distribution.npz")
    scores = data["key2"]
    avg, std = np.mean(scores), np.std(scores)
    print("--- SCORE STATISTICS ---")
    print(np.min(scores), np.max(scores), np.mean(scores), np.std(scores))
    print("---------------------\n")
    return avg - (3*std), avg + (3*std)

def main():
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

        dataset = LowHighResolutionDataset(
            h5path=f"/home-local/Frederic/evaluation-data/low-high-quality/training.hdf5",
            num_samples=None,
            transform=None,
            n_channels=1,
            num_classes=2,
            classes=["low", "high"] 
        )
        N = len(dataset)
        indices = np.arange(N)
        np.random.shuffle(indices)

        distances_to_boundary = {"low": [], "high": []}
        with torch.no_grad():
            for i in tqdm(indices, total=N):
                img, metadata = dataset[i]
                label = metadata["label"]
                torch_img = img.clone().detach().unsqueeze(0).to(DEVICE)
                latent_code = latent_encoder.forward_features(torch_img)
                numpy_code = latent_code.detach().cpu().numpy()
                d = numpy_code.dot(boundary.T) + intercept
                d = d[0][0]
                key = "low" if label == 0 else "high"
                distances_to_boundary[key].append(d)
        plot_distance_distribution(distances_to_boundary)
    else:   
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        boundary, intercept, norm = load_boundary()
        distance_min, distance_max = load_distance_distribution()
        print(f"\nMIN_DISTANCE: {distance_min} MAX_DISTANCE: {distance_max}\n")
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

        dataset = LowHighResolutionDataset(
            h5path=f"/home-local/Frederic/evaluation-data/low-high-quality/testing.hdf5",
            num_samples=None,
            transform=None,
            n_channels=1,
            num_classes=2,
            classes=["low", "high"] 
        )

        N = len(dataset)
        indices = np.arange(N)
        np.random.shuffle(indices)
        counter = 0

        all_resolutions = np.zeros((args.num_samples, args.n_steps+1))
        all_distances = np.zeros((args.num_samples, args.n_steps+1))

        for i in tqdm(indices):
            resolutions, distances = [], [] 
            img, metadata = dataset[i]
            label = metadata["label"]
            if counter >= args.num_samples:
                break 
            if args.boundary == "resolution" and label != 0: # We will only sample low poor image and move in the good res direction
                continue 

            if "imagenet" in args.weights.lower():
                img = torch.tensor(img, dtype=torch.float32).repeat(3, 1, 1).unsqueeze(0).to(DEVICE)
            else:
                img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            original = img.squeeze().detach().cpu().numpy()
            original_resolution = compute_resolution(img=original)

            latent_code = diffusion_model.latent_encoder.forward_features(img) 
            numpy_code = latent_code.detach().cpu().numpy() 
            # original_sample = diffusion_model.p_sample_loop(shape=(img.shape[0], 1, img.shape[2], img.shape[3]), cond=latent_code, progress=True) 

            resolutions.append(original_resolution)
        
            # original_sample = original_sample.squeeze().detach().cpu().numpy()
            # sample_resolution = compute_resolution(img=original_sample)
            #  resolutions.append(sample_resolution)

            samples = [original]
            distances.append(0.0)

            lerped_codes, d = linear_interpolate(latent_code=numpy_code, boundary=boundary, start_distance=0.0, end_distance=distance_max, steps=args.n_steps)

            for c, code in enumerate(lerped_codes):
                lerped_code = torch.tensor(code, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                lerped_sample = diffusion_model.p_sample_loop(shape=(img.shape[0], 1, img.shape[2], img.shape[3]), cond=lerped_code, progress=True)
                lerped_sample_numpy = lerped_sample.squeeze().detach().cpu().numpy()
                samples.append(lerped_sample_numpy)
                lerped_resolution = compute_resolution(img=lerped_sample_numpy)
                resolutions.append(lerped_resolution)
                distances.append(abs(d[c][0]))

            resolutions = np.array(resolutions)
            distances = np.array(distances)
            all_resolutions[counter] = resolutions
            all_distances[counter] = distances
            counter += 1
            save_examples(samples, distances, resolutions, counter)
        
        plot_correlation(all_resolutions, all_distances)



if __name__=="__main__":
    main() 