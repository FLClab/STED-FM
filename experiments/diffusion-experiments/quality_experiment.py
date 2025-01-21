import numpy as np 
import matplotlib.pyplot as plt 
from QualityNet.networks import NetTrueFCN 
import argparse 
import torch 
from torch import nn
from diffusion_models.diffusion.ddpm_lightning import DDPM 
from diffusion_models.diffusion.denoising.unet import UNet 
from tqdm import trange, tqdm 
import copy 
import random 
import os
from attribute_datasets import get_dataset, OptimQualityDataset
import sys
import glob
sys.path.insert(0, "../")
from DEFAULTS import BASE_PATH, COLORS
from model_builder import get_pretrained_model_v2

parser = argparse.ArgumentParser()
parser.add_argument("--latent-encoder", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--timesteps", type=int, default=1000)
parser.add_argument("--boundary", type=str, default="quality")
parser.add_argument("--num-samples", type=int, default=20)
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
  """Manipulates the given latent code with respect to a particular boundary.

  Basically, this function takes a latent code and a boundary as inputs, and
  outputs a collection of manipulated latent codes. For example, let `steps` to
  be 10, then the input `latent_code` is with shape [1, latent_space_dim], input
  `boundary` is with shape [1, latent_space_dim] and unit norm, the output is
  with shape [10, latent_space_dim]. The first output latent code is
  `start_distance` away from the given `boundary`, while the last output latent
  code is `end_distance` away from the given `boundary`. Remaining latent codes
  are linearly interpolated.

  Input `latent_code` can also be with shape [1, num_layers, latent_space_dim]
  to support W+ space in Style GAN. In this case, all features in W+ space will
  be manipulated same as each other. Accordingly, the output will be with shape
  [10, num_layers, latent_space_dim].

  NOTE: Distance is sign sensitive.

  Args:
    latent_code: The input latent code for manipulation.
    boundary: The semantic boundary as reference.
    start_distance: The distance to the boundary where the manipulation starts.
      (default: -3.0)
    end_distance: The distance to the boundary where the manipulation ends.
      (default: 3.0)
    steps: Number of steps to move the latent code from start position to end
      position. (default: 10)
  """
  assert (latent_code.shape[0] == 1 and boundary.shape[0] == 1 and
          len(boundary.shape) == 2 and
          boundary.shape[1] == latent_code.shape[-1])

  
  linspace = np.linspace(start_distance, end_distance, steps)
  if len(latent_code.shape) == 2:
    linspace = linspace - ((latent_code.dot(boundary.T)) + intercept)
    linspace = linspace.reshape(-1, 1).astype(np.float32)
    return latent_code + linspace * boundary, linspace
  if len(latent_code.shape) == 3:
    linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
    return latent_code + linspace * boundary.reshape(1, 1, -1), linspace
  raise ValueError(f'Input `latent_code` should be with shape '
                   f'[1, latent_space_dim] but {latent_code.shape} was received.')

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

def load_boundary() -> np.ndarray:
    print(f"--- Loading boundary trained from {args.weights} embeddings ---")
    data = np.load(f"./{args.boundary}-experiment/boundaries/{args.weights}_{args.boundary}_boundary.npz")
    boundary, intercept, norm = data["boundary"], data["intercept"], data["norm"]
    return boundary, intercept, norm

def load_distance_distribution(weight: str = args.weights) -> np.ndarray:
    data = np.load(f"./{args.boundary}-experiment/distributions/{weight}-quality-distance_distribution.npz")
    scores = data["key2"]
    avg, std = np.mean(scores), np.std(scores)
    max_distance = np.max(scores) 
    return avg - (5*std), max_distance * 2

def load_quality_net() -> nn.Module:
    quality_net = NetTrueFCN()
    quality_checkpoint = torch.load(f"./QualityNet/trained_models/actin/qualitynet.pth")
    quality_net.load_state_dict(quality_checkpoint["model_state_dict"])
    return quality_net

def infer_quality(img: torch.Tensor, quality_net: nn.Module) -> float:
    if img.shape[1] == 3:
        img = img[:, [0], :, :]
    quality_net.eval()
    with torch.no_grad():
        score = quality_net(img)
    return score.item()

def plot_correlation(all_scores, all_distances, original_scores):
    scores = np.mean(all_scores, axis=0)
    distances = np.mean(all_distances, axis=0)
    distances = [round(d, 3) for d in distances]
    err = np.std(all_scores, axis=0)
    np.savez(f"./{args.boundary}-experiment/correlation/{args.weights}-quality-correlation.npz", 
             all_scores=all_scores, all_distances=all_distances, original_scores=original_scores,
             scores=scores, distances=distances, err=err)
    fig = plt.figure(figsize=(5,5))
    plt.plot(distances, scores, c="black")
    plt.fill_between(distances, scores-err, scores+err, color="black", alpha=0.2)
    plt.xlabel("Distance")
    plt.ylabel("Score")
    plt.title("Distance vs Score")
    fig.savefig(f"./{args.boundary}-experiment/correlation/{args.weights}-quality-correlation.png", bbox_inches="tight", dpi=1200)
    plt.close()

    fig = plt.figure(figsize=(5,5))
    cmap = plt.get_cmap("RdPu")
    norm = plt.Normalize(vmin=min(original_scores), vmax=max(original_scores))
    for i in range(all_scores.shape[0]):
        oscore = original_scores[i]
        plt.scatter(all_distances[i], all_scores[i], c=[oscore]*all_distances[i].shape[0], cmap=cmap, norm=norm, edgecolors="black")
    plt.xlabel("Distance traveled")
    plt.ylabel("Score gain")
    fig.savefig(f"./{args.boundary}-experiment/correlation/{args.weights}-quality-scatter.png", bbox_inches="tight", dpi=1200)
    plt.close()


def save_examples(samples, distances, scores, raw_scores, index):
    print("--- Saving examples ---")
    N = len(samples)
    fig, axs = plt.subplots(1, N, figsize=(10, 5))
    for i, (s, d, sc, rs) in enumerate(zip(samples, distances, scores, raw_scores)):
        print(f"... {i} / {N} ...")
        if s.shape[0] == 3:
            s = s[0, :, :]
        axs[i].imshow(s, cmap='hot', vmin=0.0, vmax=1.0)
        axs[i].set_title("Distance: {:.2f}\nScore: {:.2f}\n({:.2f})".format(d, sc, rs))
        axs[i].axis("off")
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
    fig.savefig(f"./{args.boundary}-experiment/examples/{args.weights}-image_{index}.pdf", dpi=1200, bbox_inches='tight')
    plt.close(fig)

def plot_results():
    files = glob.glob(f"./{args.boundary}-experiment/correlation/**-quality-correlation.npz")
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for f in files:
        w = f.split("/")[-1].split("-")[0]
        weight = w.replace("MAE_SMALL_", "")
        _, distance_max = load_distance_distribution(weight=w)
        data = np.load(f)
        scores, distances, original_scores = data["all_scores"], data["all_distances"], data["original_scores"]
        original_scores = np.array(original_scores)
        mask = np.where(original_scores >= 0.10)
        original_scores = original_scores[mask]
        scores = scores[mask]
        distances = distances[mask]
        mean_scores = np.mean(scores, axis=0)
        #lower_bounds, upper_bounds = compute_confidence_intervals(scores)
        x = np.linspace(0.0, distance_max, distances[0].shape[0])
        ax.plot(x, mean_scores, c=COLORS[weight], label=weight, marker='o')
        #ax.fill_between(x, lower_bounds, upper_bounds, color=COLORS[weight], alpha=0.2)
        # axs[0].fill_between(x, lower_bounds, upper_bounds, color=COLORS[weight], alpha=0.2)
        # for i in range(scores.shape[0]):
        #     axs[1].scatter(distances[i], scores[i], c=[COLORS[weight]]*distances[i].shape[0], edgecolors="black")
    # axs[0].set_xlabel("Distance from boundary")
    # axs[0].set_ylabel("Score gain")
    # axs[0].legend()
    # axs[1].set_xlabel("Distance from original embedding")
    # axs[1].set_ylabel("Score gain")
    ax.set_xlabel("Distance from boundary")
    ax.set_ylabel("Score gain")
    ax.legend()
    fig.savefig(f"./{args.boundary}-experiment/correlation/quality-correlation.pdf", bbox_inches="tight", dpi=1200)
    plt.close()

def plot_distance_distribution(distances_to_boundary: dict):
    key1, key2 = list(distances_to_boundary.keys())
    np.savez(f"./{args.boundary}-experiment/distributions/{args.weights}-quality-distance_distribution.npz", key1=distances_to_boundary[key1], key2=distances_to_boundary[key2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(distances_to_boundary["low"], bins=100, alpha=0.5, color='fuchsia', label="Low")
    ax.hist(distances_to_boundary["high"], bins=100, alpha=0.5, color='dodgerblue', label="High")
    ax.axvline(0.0, color='black', linestyle='--', label="Decision boundary")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.savefig(f"./{args.boundary}-experiment/distributions/{args.weights}-quality-distance_distribution.pdf", dpi=1200, bbox_inches="tight")
    plt.close(fig)

def cumulative_regret() -> None:
    files = glob.glob(f"./{args.boundary}-experiment/correlation/**-quality-correlation.npz")
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for f in files:
        w = f.split("/")[-1].split("-")[0]
        weight = w.replace("MAE_SMALL_", "")
        data = np.load(f)
        distance_min, distance_max = load_distance_distribution(weight=w)
        all_scores, all_distances, original_scores = data["all_scores"], data["all_distances"], data["original_scores"]

        regret_per_image = []
        image_distances = []
        for i in range(all_scores.shape[0]):
            res = all_scores[i]
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
        ax1.plot(x, regret_per_image, c=COLORS[weight], label=weight, marker='o')
        ax2.scatter(image_distances, regret_per_image, c=COLORS[weight], label=weight)
    ax1.set_xlabel("Image index")
    ax1.set_ylabel("Cumulative regret (quality score)")
    ax2.set_xlabel("Distance from boundary")
    ax2.set_ylabel("Image regret (quality score)")
    ax1.legend()
    fig.savefig(f"./{args.boundary}-experiment/correlation/{args.weights}-regret.pdf", bbox_inches="tight", dpi=1200)
    plt.close()


def main():
    if args.figure:
        plot_results() 
        cumulative_regret()
    elif args.sanity_check:
        np.random.seed(42)
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

        dataset = OptimQualityDataset(
            data_folder="/home-local/Frederic/evaluation-data/optim_train",
            num_samples={"actin": None},
            classes=['actin'],
            high_score_threshold=0.70,
            low_score_threshold=0.60,
            n_channels=1
        )
        N = len(dataset)

        distances_to_boundary = {"low": [], "high": []}
        with torch.no_grad():
            for i in tqdm(range(N), total=N):
                img, metadata = dataset[i]
                score = metadata["score"]
                label = metadata["label"]
               
                if "imagenet" in args.weights.lower():
                    torch_img = img.clone().detach().repeat(3, 1, 1).unsqueeze(0).to(DEVICE)
                else:
                    torch_img = img.clone().detach().unsqueeze(0).to(DEVICE)

                latent_code = latent_encoder.forward_features(torch_img)
                numpy_code = latent_code.detach().cpu().numpy()
                
                d = numpy_code.dot(boundary.T) + intercept
                d = d[0][0]
                key = "low" if label == 0 else "high"
                distances_to_boundary[key].append(d)
        plot_distance_distribution(distances_to_boundary)

    else:
        np.random.seed(42)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        boundary, intercept, _ = load_boundary()
        distance_min, distance_max = load_distance_distribution()
        print(f"--- Moving from 0.0 to {distance_max} ---")

        quality_net = load_quality_net()
        quality_net.to(DEVICE)

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
        diffusion_model.eval()


        dataset = OptimQualityDataset(
            data_folder="/home-local/Frederic/evaluation-data/optim-data",
            num_samples={"actin": None},
            classes=['actin'],
            high_score_threshold=0.70,
            low_score_threshold=0.60,
            n_channels=1
        )
        N = len(dataset)
        indices = np.arange(N)
        print(f"Dataset size: {N}")
        np.random.shuffle(indices)
        counter = 0 

        with torch.no_grad():
            all_scores = np.zeros((args.num_samples, args.n_steps+1))
            all_distances = np.zeros((args.num_samples, args.n_steps+1))
            original_scores = []
            for i in tqdm(indices, total=N):
                scores, distances, raw_scores = [], [], []
                if counter >= args.num_samples:
                    break
                img, metadata = dataset[i]
                label = metadata["label"]
                score = metadata["score"]
                if label != 0 or score > 0.50:
                    continue

                if "imagenet" in args.weights.lower():
                    img = img.clone().detach().repeat(3, 1, 1).unsqueeze(0).to(DEVICE)
                else:
                    img = img.clone().detach().unsqueeze(0).to(DEVICE)
                original = img.squeeze().detach().cpu().numpy()

                latent_code = diffusion_model.latent_encoder.forward_features(img) 
                
                numpy_code = latent_code.detach().cpu().numpy() 
                distance_score = numpy_code.dot(boundary.T) + intercept
                d = distance_score[0][0]
            
                # original_sample = diffusion_model.p_sample_loop(shape=(img.shape[0], 1, img.shape[2], img.shape[3]), cond=latent_code, progress=True) 

                original_score = infer_quality(img, quality_net)
                scores.append(original_score - original_score)
                raw_scores.append(original_score)
                original_scores.append(original_score)
                # scores.append(infer_quality(original_sample, quality_net) - original_score)

                # original_sample = original_sample.squeeze().detach().cpu().numpy()
                samples = [original]

                distances.append(0.0)

                lerped_codes, d = linear_interpolate(latent_code=numpy_code, boundary=boundary, intercept=intercept, start_distance=0.0, end_distance=distance_max, steps=args.n_steps)

                for c, code in enumerate(lerped_codes):
                    lerped_code = torch.tensor(code, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    lerped_sample = diffusion_model.p_sample_loop(shape=(img.shape[0], 1, img.shape[2], img.shape[3]), cond=lerped_code, progress=True)
                    lerped_sample_numpy = lerped_sample.squeeze().detach().cpu().numpy()
                    samples.append(lerped_sample_numpy)
                    curr_score = infer_quality(lerped_sample, quality_net)
                    scores.append(curr_score - original_score)
                    raw_scores.append(curr_score)
                    distances.append(abs(d[c][0]))
            
                scores = np.array(scores)
                distances = np.array(distances)
                all_scores[counter] = scores
                all_distances[counter] = distances
                counter += 1
                save_examples(samples, distances, scores, raw_scores, counter)

        print(all_scores.shape, all_distances.shape, len(original_scores))
        plot_correlation(all_scores, all_distances, original_scores)


if __name__=="__main__":
    main()

