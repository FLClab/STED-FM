import numpy as np
import matplotlib.pyplot as plt 
import argparse 
import torch 
from torch import nn 
from diffusion_models.diffusion.ddpm_lightning import DDPM 
from diffusion_models.diffusion.denoising.unet import UNet 
from tqdm import tqdm 
from typing import Union
from attribute_datasets import LowHighResolutionDataset
import sys
from banditopt.objectives import Resolution 
import glob
from stedfm.DEFAULTS import BASE_PATH, COLORS, MARKERS
from stedfm.model_builder import get_pretrained_model_v2

parser = argparse.ArgumentParser()
parser.add_argument("--latent-encoder", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--timesteps", type=int, default=1000)
parser.add_argument("--boundary", type=str, default="resolution")
parser.add_argument("--num-samples", type=int, default=12)
parser.add_argument("--ckpt-path", type=str, default="/home-local/Frederic/baselines/DiffusionModels/latent-guidance")
parser.add_argument("--figure", action="store_true")
parser.add_argument("--sanity-check", action="store_true")
parser.add_argument("--n-steps", type=int, default=6)
args = parser.parse_args()


def denormalize(img: Union[np.ndarray, torch.Tensor], mu: float = 0.010903545655310154, std: float = 0.03640301525592804) -> Union[np.ndarray, torch.Tensor]:
    """
    Denormalizes an image. Note that the parameters mu and sigma seem hard-coded but they have been computed from the training sets and can be found
    in the attribute_datasets.py file.
    """
    return img * std + mu

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
                       norm,
                       start_distance=-4.0,
                       end_distance=4.0,
                       steps=8):
    assert (latent_code.shape[0] == 1 and boundary.shape[0] == 1 and
          len(boundary.shape) == 2 and
          boundary.shape[1] == latent_code.shape[-1])

    img_distance = latent_code.dot(boundary.T) + intercept
    end_distance = end_distance - img_distance
    linspace = np.linspace(start_distance, end_distance, steps)[1:]
    # linspace = np.linspace(img_distance, end_distance, steps)
    if len(latent_code.shape) == 2:
        # linspace = linspace - (latent_code.dot(boundary.T) + intercept)
        linspace = linspace.reshape(-1, 1).astype(np.float32)
        return latent_code + linspace * boundary * norm, linspace, img_distance[0][0]
    if len(latent_code.shape) == 3:
        linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
        return latent_code + linspace * boundary.reshape(1, 1, -1), linspace
    raise ValueError(f'Input `latent_code` should be with shape '
                    f'[1, latent_space_dim] but {latent_code.shape} was received.')

def load_boundary() -> np.ndarray:
    print(f"--- Loading boundary trained from {args.weights} embeddings ---")
    data = np.load(f"./{args.boundary}-experiment/boundaries/{args.weights}_{args.boundary}_boundary.npz")
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
       axs[i].imshow(s, cmap="hot")# , vmin=0.0, vmax=1.0)
       axs[i].set_title(f"Distance: {d:.2f}\nRes: {r:.2f}")
       axs[i].axis("off")
   plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
   fig.savefig(f"./{args.boundary}-experiment/examples/{args.weights}-image_{index}.pdf", dpi=1200, bbox_inches="tight")
   plt.close(fig)

def plot_distance_distribution(distances_to_boundary: dict):
    key1, key2 = list(distances_to_boundary.keys())
    np.savez(f"./{args.boundary}-experiment/distributions/{args.weights}-resolution-distance_distribution.npz", key1=distances_to_boundary[key1], key2=distances_to_boundary[key2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(distances_to_boundary["low"], bins=50, alpha=0.5, color='fuchsia', label="Low")
    ax.hist(distances_to_boundary["high"], bins=50, alpha=0.5, color='dodgerblue', label="High")
    ax.axvline(0.0, color='black', linestyle='--', label="Decision boundary")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.savefig(f"./{args.boundary}-experiment/distributions/{args.weights}-resolution-distance_distribution.pdf", dpi=1200, bbox_inches="tight")
    plt.close(fig)

def plot_resolution_distributions(resolutions: dict) -> None:
    key1, key2 = list(resolutions.keys())
    np.savez(f"./{args.boundary}-experiment/distributions/{args.weights}-resolution-resolution_distribution.npz", key1=resolutions[key1], key2=resolutions[key2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    k1_mean, k2_mean = np.mean(resolutions[key1]), np.mean(resolutions[key2])
    res1 = np.array(resolutions[key1])
    mask1 = np.where(res1 < 300)
    res1 = res1[mask1] 
    res2 = np.array(resolutions[key2])
    mask2 = np.where(res2 < 300)
    res2 = res2[mask2]
    k1_mean, k2_mean = np.mean(res1), np.mean(res2)
    ax.hist(res1, bins=50, alpha=0.5, color='fuchsia', label="Low")
    ax.hist(res2, bins=50, alpha=0.5, color='dodgerblue', label="High")

    ax.axvline(k1_mean, color='fuchsia', linestyle='--', label="Low mean")
    ax.text(k1_mean -5, ax.get_ylim()[1]*0.9, f'{k1_mean:.2f}', color='fuchsia', ha='center')
    ax.axvline(k2_mean, color='dodgerblue', linestyle='--', label="High mean")
    ax.text(k2_mean - 5, ax.get_ylim()[1]*0.9, f'{k2_mean:.2f}', color='dodgerblue', ha='center')
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.savefig(f"./{args.boundary}-experiment/distributions/{args.weights}-resolution-resolution_distribution.pdf", dpi=1200, bbox_inches="tight")
    plt.close(fig)

def plot_correlation(all_resolutions, all_distances, original_resolutions):
    resolutions = np.mean(all_resolutions, axis=0)
    distances = np.mean(all_distances, axis=0)
    distances = [round(d, 3) for d in distances]
    err = np.std(all_resolutions, axis=0)
    np.savez(f"./{args.boundary}-experiment/correlation/{args.weights}-resolution-correlation.npz", 
             all_resolutions=all_resolutions, all_distances=all_distances, original_resolutions=original_resolutions,
             resolutions=resolutions, distances=distances, err=err)
    fig = plt.figure(figsize=(5,5))
    plt.plot(distances, resolutions, c="black")
    plt.fill_between(distances, resolutions-err, resolutions+err, color="black", alpha=0.2)
    plt.xlabel("Distance")
    plt.ylabel("Resolution")
    plt.title("Distance vs Resolution")
    fig.savefig(f"./{args.boundary}-experiment/correlation/{args.weights}-resolution-correlation.png", bbox_inches="tight", dpi=1200)
    plt.close()

    fig = plt.figure(figsize=(5,5))
    cmap = plt.get_cmap("RdPu")
    norm = plt.Normalize(vmin=min(original_resolutions), vmax=max(original_resolutions))
    for i in range(all_resolutions.shape[0]):
        oscore = original_resolutions[i]
        plt.scatter(all_distances[i], all_resolutions[i], c=[oscore]*all_distances[i].shape[0], cmap=cmap, norm=norm, edgecolors="black")
    plt.xlabel("Distance traveled")
    plt.ylabel("Resolution")
    fig.savefig(f"./{args.boundary}-experiment/correlation/{args.weights}-resolution-scatter.png", bbox_inches="tight", dpi=1200)
    plt.close()


def plot_results():
    files = glob.glob(f"./{args.boundary}-experiment/correlation/**-resolution-correlation.npz")
    # fig, axs = plt.subplots(1,2, figsize=(10,5))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for f in files:
        w = f.split("/")[-1].split("-")[0]
        weight = w.replace("MAE_SMALL_", "")
        distance_min, distance_max = load_distance_distribution(weight=w)
        data = np.load(f)
        resolutions, distances, err = data["resolutions"], data["distances"], data["err"]
        all_resolutions, all_distances, original_resolutions = data["all_resolutions"], data["all_distances"], data["original_resolutions"]
        x = np.linspace(0.0, distance_max, distances.shape[0])
        # axs[0].plot(x, resolutions, c=COLORS[weight], label=weight)
        lower_bounds, upper_bounds = compute_confidence_intervals(all_resolutions)
        # axs[0].fill_between(x, lower_bounds, upper_bounds, color=COLORS[weight], alpha=0.2)
        ax.plot(x, resolutions, c=COLORS[weight], label=weight, marker=MARKERS[weight])
        # for i in range(all_resolutions.shape[0]):
        #     axs[1].scatter(all_distances[i], all_resolutions[i], c=[COLORS[weight]]*all_distances[i].shape[0], edgecolors="black")
    # axs[0].legend()
    # axs[0].set_xlabel("Distance from boundary")
    # axs[0].set_ylabel("Resolution")
    # axs[1].set_xlabel("Distance from embedding")
    # axs[1].set_ylabel("Resolution")
    ax.set_xlabel("Distance from boundary")
    ax.set_ylabel("Resolution")
    ax.legend()
    fig.savefig(f"./{args.boundary}-experiment/correlation/resolution-correlation.pdf", bbox_inches="tight", dpi=1200)
    plt.close()


def cumulative_regret() -> None:
    files = glob.glob(f"./{args.boundary}-experiment/correlation/**-resolution-correlation.npz")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for f in files:
        w = f.split("/")[-1].split("-")[0]
        weight = w.replace("MAE_SMALL_", "")
        data = np.load(f)
        distance_min, distance_max = load_distance_distribution(weight=w)
        all_resolutions, all_distances, original_resolutions = data["all_resolutions"], data["all_distances"], data["original_resolutions"]
        resolutions, distances = np.mean(all_resolutions, axis=0), np.mean(all_distances, axis=0)

        regret_per_image = []
        image_distances = []
        for i in range(all_resolutions.shape[0]):
            res = all_resolutions[i]
            d = all_distances[i][1]
            image_distances.append(d)
            image_regret = 0.0
            for j in range(1, res.shape[0]):
                diff = res[j] - res[j-1] 
                if diff >= 0: 
                    image_regret += diff
            regret_per_image.append(image_regret)
            
        regret_per_image = np.cumsum(regret_per_image)
        x = np.arange(len(regret_per_image))
        ax.plot(x, regret_per_image, c=COLORS[weight], label=weight, marker=MARKERS[weight])

    # ax.set_yscale('log')
    ax.set_xlabel("Image index")
    ax.set_ylabel("Cumulative regret (nm)")
    ax.legend()
    fig.savefig(f"./{args.boundary}-experiment/correlation/resolution-regret.pdf", bbox_inches="tight", dpi=1200)
    plt.close()


def load_distance_distribution(weight: str = args.weights) -> np.ndarray:
    data = np.load(f"./{args.boundary}-experiment/distributions/{weight}-resolution-distance_distribution.npz")
    scores = data["key2"]
    avg, std = np.mean(scores), np.std(scores)
    min_distance, max_distance = 0, np.max(scores)
    # max_distance = max_distance * 2 if "imagenet" not in args.weights.lower() else max_distance
    return min_distance, max_distance

def main():
    if args.figure:
        plot_results()
        cumulative_regret()
    elif args.sanity_check:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        channels = 3 if "imagenet" in args.weights.lower() else 1
        boundary, intercept, norm = load_boundary()
        latent_encoder, model_config = get_pretrained_model_v2(
            name=args.latent_encoder,
            weights=args.weights,
            path=None,
            mask_ratio=0.0, 
            pretrained=True if "imagenet" in args.weights.lower() else False,
            in_channels=channels,
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
            n_channels=channels,
            num_classes=2,
            classes=["low", "high"] 
        )
        N = len(dataset)
        indices = np.arange(N)
        np.random.shuffle(indices)

        distances_to_boundary = {"low": [], "high": []}
        train_resolutions = {"low": [], "high": []}
        with torch.no_grad():
            for i in tqdm(indices, total=N):
                img, metadata = dataset[i]
                label = metadata["label"]
                
                torch_img = img.clone().unsqueeze(0).to(DEVICE)
                latent_code = latent_encoder.forward_features(torch_img)
                numpy_code = latent_code.detach().cpu().numpy()
                d = numpy_code.dot(boundary.T) + intercept
                d = d[0][0]
                key = "low" if label == 0 else "high"
                distances_to_boundary[key].append(d)
                img_numpy = img.squeeze().detach().cpu().numpy()
                resolution = compute_resolution(img_numpy)
                train_resolutions[key].append(resolution)
        plot_distance_distribution(distances_to_boundary)
        plot_resolution_distributions(train_resolutions)
    else:   
        np.random.seed(42)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        boundary, intercept, norm = load_boundary()
        channels = 3 if "imagenet" in args.weights.lower() else 1
        distance_min, distance_max = load_distance_distribution()
        print(f"--- Moving from 0.0 to {distance_max} ---")
        latent_encoder, model_config = get_pretrained_model_v2(
            name=args.latent_encoder,
            weights=args.weights,
            path=None, 
            mask_ratio=0.0,
            pretrained=True if "imagenet" in args.weights.lower() else False,
            in_channels=channels,
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
        diffusion_model.load_state_dict(ckpt["state_dict"], strict=True)
        diffusion_model.to(DEVICE)
        diffusion_model.eval()

        dataset = LowHighResolutionDataset(
            h5path=f"/home-local/Frederic/evaluation-data/low-high-quality/testing.hdf5",
            num_samples=None,
            transform=None,
            n_channels=channels,
            num_classes=2,
            classes=["low", "high"] 
        )

        N = len(dataset)
        indices = np.arange(N)
        np.random.shuffle(indices)
        counter = 0

        all_resolutions = np.zeros((args.num_samples, args.n_steps+1))
        all_distances = np.zeros((args.num_samples, args.n_steps+1))
        original_resolutions = []

        with torch.no_grad():
            for i in tqdm(indices):
                resolutions, distances = [], [] 
                img, metadata = dataset[i]
                label = metadata["label"]
                if counter >= args.num_samples:
                    break 
                if args.boundary == "resolution" and label != 0: # We will only sample low poor image and move in the good res direction
                    continue 

                
                img = img.clone().unsqueeze(0).to(DEVICE)
                latent_code = diffusion_model.latent_encoder.forward_features(img) 
                
                if "imagenet" in args.weights.lower():
                    img = img[:, [0], :, :]

                original = img.squeeze().detach().cpu().numpy()
                original_resolution = compute_resolution(img=original)
                original_resolutions.append(original_resolution)

    
                numpy_code = latent_code.detach().cpu().numpy() 
                # original_sample = diffusion_model.p_sample_loop(shape=(img.shape[0], 1, img.shape[2], img.shape[3]), cond=latent_code, progress=True) 

                resolutions.append(original_resolution)
            
                # original_sample = original_sample.squeeze().detach().cpu().numpy()
                # sample_resolution = compute_resolution(img=original_sample)
                #  resolutions.append(sample_resolution)

                samples = [original]
                # distances.append(0.0)

                lerped_codes, d, img_distance = linear_interpolate(latent_code=numpy_code, boundary=boundary, intercept=intercept, norm=norm, start_distance=0.0, end_distance=distance_max, steps=args.n_steps + 1)

                distances.append(img_distance)

                for c, code in enumerate(lerped_codes):
                    lerped_code = torch.tensor(code, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    lerped_sample = diffusion_model.p_sample_loop(shape=(img.shape[0], 1, img.shape[2], img.shape[3]), cond=lerped_code, progress=True)
                    lerped_sample = denormalize(lerped_sample)
                    lerped_sample_numpy = lerped_sample.squeeze().detach().cpu().numpy()
                    samples.append(lerped_sample_numpy)
                    lerped_resolution = compute_resolution(img=lerped_sample_numpy)
                    resolutions.append(lerped_resolution)
                    # print(d[c][0])
                    # print(d[c][0].shape)
                    # print(img_distance + d[c][0])
                    distances.append(img_distance + d[c][0])

                resolutions = np.array(resolutions)
                distances = np.array(distances)
                all_resolutions[counter] = resolutions
                all_distances[counter] = distances
                counter += 1
                save_examples(samples, distances, resolutions, counter)
        print(all_resolutions.shape, all_distances.shape, len(original_resolutions))
        plot_correlation(all_resolutions, all_distances, original_resolutions)



if __name__=="__main__":
    main() 