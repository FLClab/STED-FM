import pickle
import numpy as np
import tifffile
import os
import torch

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
from typing import Union
from scipy.spatial import distance

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

def denormalize(img: Union[np.ndarray, torch.Tensor], mu: float = 0.0695771782959453, std: float = 0.12546228631005282) -> np.ndarray:
    return img * std + mu

def load_svm(args):
    with open(f"./{args.boundary}-experiment/boundaries/{args.weights}_{args.boundary}_svm.pkl", "rb") as f:
        svm = pickle.load(f)
    return svm

def load_boundary(args) -> np.ndarray:
    print(f"--- Loading boundary trained from {args.weights} embeddings ---")
    data = np.load(f"./{args.boundary}-experiment/boundaries/{args.weights}_{args.boundary}_boundary.npz")
    boundary, intercept, norm = data["boundary"], data["intercept"], data["norm"]
    return boundary, intercept, norm

def load_distance_distribution(args) -> np.ndarray:
    data = np.load(f"./{args.boundary}-experiment/distributions/{args.weights}-{args.boundary}-distance_distribution.npz")
    scores = data[args.origin]
    start_distance = np.mean(scores)

    scores = data[args.direction]
    end_distance = np.mean(scores)

    return start_distance, end_distance

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

    # Compute distance from the latent code to the hyperplane
    # This corresponds to the SVM decision function value
    latent_code_distance_from_boundary = latent_code.dot(boundary.T) + intercept

    # Linspace of distances along the normal vector; this is relative to the hyperplane
    linspace = np.linspace(start_distance, end_distance, steps)# [1:]
    if len(latent_code.shape) == 2:
        linspace = linspace - latent_code_distance_from_boundary
        linspace = linspace.reshape(-1, 1).astype(np.float32) 

        # Generate new latent codes by moving along the boundary normal vector
        # This recenters the movement around boundary
        latent_codes = latent_code + linspace * boundary 

        # Compute distances to boundary for each new latent code        
        distances = latent_codes.dot(boundary.T) + intercept 
        distances = distances.flatten()

        return latent_codes, distances, latent_code_distance_from_boundary[0][0]
    if len(latent_code.shape) == 3:
        linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
        return latent_code + linspace * boundary.reshape(1, 1, -1), linspace
    raise ValueError(f'Input `latent_code` should be with shape '
                    f'[1, latent_space_dim] but {latent_code.shape} was received.')

def plot_distance_distribution(args, distances_to_boundary: dict):

    os.makedirs(f"./{args.boundary}-experiment/distributions", exist_ok=True)
    np.savez(f"./{args.boundary}-experiment/distributions/{args.weights}-{args.boundary}-distance_distribution.npz", **distances_to_boundary)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(distances_to_boundary[args.origin], bins=100, alpha=0.5, color='fuchsia', label="Low")
    ax.hist(distances_to_boundary[args.direction], bins=100, alpha=0.5, color='dodgerblue', label="High")
    ax.axvline(0.0, color='black', linestyle='--', label="Decision boundary")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.savefig(f"./{args.boundary}-experiment/distributions/{args.weights}-{args.boundary}-distance_distribution.pdf", dpi=1200, bbox_inches="tight")
    plt.close(fig)

def save_raw_images(args, samples, titles, index):
    os.makedirs(f"./{args.boundary}-experiment/examples/raw", exist_ok=True)
    os.makedirs(f"./{args.boundary}-experiment/examples/raw-tif", exist_ok=True)
    
    cmap = plt.get_cmap('hot')
    norm = Normalize(vmin=0.0, vmax=1.0, clip=True)

    for i, (s, d) in enumerate(zip(samples, titles)):
        if s.shape[0] == 3:
            s = s[0, :, :]

        tifffile.imwrite(
            f"./{args.boundary}-experiment/examples/raw-tif/{args.weights}-image_{index}_to{args.direction}_{i}.tif",
            s.astype(np.float32)
        )

        img = Image.fromarray((cmap(norm(s)) * 255).astype(np.uint8))
        img.save(f"./{args.boundary}-experiment/examples/raw/{args.weights}-image_{index}_to{args.direction}_{i}.png")

def save_examples(args, samples, titles, index):
    os.makedirs(f"./{args.boundary}-experiment/examples", exist_ok=True)

    N = len(samples)
    fig, axs = plt.subplots(1, N, figsize=(10, 5))
    for i, (s, title) in enumerate(zip(samples, titles)):
        if s.shape[0] == 3:
            s = s[0, :, :]
        axs[i].imshow(s, cmap='hot', vmin=0.0, vmax=1.0)
        axs[i].set_title(title)
        axs[i].axis("off")
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
    fig.savefig(f"./{args.boundary}-experiment/examples/{args.weights}-image_{index}_to{args.direction}.pdf", dpi=1200, bbox_inches='tight')
    plt.close(fig)