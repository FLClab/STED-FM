import numpy as np
import matplotlib.pyplot as plt 
import argparse 
from models.diffusion.diffusion_model import DDPM 
from models.diffusion.denoising.unet import UNet 
import torch 
from tqdm import trange, tqdm 
import copy 
import sys 
import random 
from utils import denormalize
import os
from quality_dataset import OptimQualityDataset 
from DEFAULTS import BASE_PATH 
from model_builder import get_pretrained_model

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset-path", type=str, default="./Datasets/FLCDataset/baselines/dataset.tar")
parser.add_argument("--model", type=str, default="mae-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--timesteps", type=int, default=1000)
parser.add_argument("--dataset", type=str, default="STED")
parser.add_argument("--checkpoint", type=str, default='./model_checkpoints/DiffusionModels/mae-small_STED/checkpoint-139.pth')
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--num-samples", type=int, default=10)
parser.add_argument("--sampling", type=str, default="ddpm")
args = parser.parse_args()

def linear_interpolate(latent_code,
                       boundary,
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
  print(latent_code.shape, boundary.shape)
  if len(latent_code.shape) == 2:
    print(f"linspace.shape: {linspace.shape}")
    linspace = linspace - latent_code.dot(boundary.T)
    print(f"linspace - latent_code.dot(boundary.T).shape: {linspace.shape}")
    linspace = linspace.reshape(-1, 1).astype(np.float32)
    print(f"linspace.reshape(-1, 1).shape: {linspace.shape}")
    print(f"linspace * boundary.shape: {(linspace * boundary).shape}")
    print(f"latent_code + linspace * boundary.shape: {(latent_code + linspace * boundary).shape}")
    return latent_code + linspace * boundary, linspace
  if len(latent_code.shape) == 3:
    linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
    return latent_code + linspace * boundary.reshape(1, 1, -1), linspace
  raise ValueError(f'Input `latent_code` should be with shape '
                   f'[1, latent_space_dim] but {latent_code.shape} was received.')


def main():
    boundary = np.load("./lerp-results/optim_quality_boundary.npz")["boundary"]
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 
    latent_encoder, model_config = get_pretrained_model(
        name=args.model,
        weights=args.weights,
        path=None, 
        mask_ratio=0.0,
        pretrained=False,
        in_channels=1,
        as_classifier=True,
        blocks="all",
        num_classes=4
    ) 
    denoising_model = UNet(
        dim=64, 
        channels=1,
        cond_dim=model_config.dim,
        dim_mults=(1,2,4),
        condition_type="latent",
        num_classes=4,
    )
    model = DDPM(
        denoising_model=denoising_model,
        timesteps=args.timesteps,
        beta_schedule="linear",
        condition_type="latent",
        latent_encoder=latent_encoder
    )

    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE)

    dataset = OptimQualityDataset(
        "./lerp-results/optim-data",
        num_samples={"actin": None, "tubulin": None, "CaMKII_Neuron": None, "PSD95_Neuron": None},
        apply_filter=True,
        classes=['actin', 'tubulin', 'CaMKII_Neuron', 'PSD95_Neuron'],
        high_score_threshold=0.70,
        low_score_threshold=0.60
    )
    N = len(dataset)

    indices = np.arange(N)
    np.random.shuffle(indices)
    counter = 0

    for i in indices:
      if counter >= args.num_samples:
        print("--- Done ---")
        exit() 
      img, metadata = dataset[i]
      score = metadata["score"] 
      if score > 0.60:
        continue
      img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)
      original = img.squeeze().detach().cpu().numpy()


      latent_code = model.latent_encoder.forward_features(img)
      numpy_code = latent_code.detach().cpu().numpy()
      original_sample = model.p_sample_loop(shape=img.shape, cond=latent_code, progress=True) 
      original_sample = original_sample.squeeze().detach().cpu().numpy()
      all_samples = []
      
      lerped_codes, distances = linear_interpolate(latent_code=numpy_code, boundary=boundary)
      print("\n")
      for l in lerped_codes:
        print(l.shape)

      exit()
      print(distances)
      for c, code in enumerate(lerped_codes):
        lerped_code = torch.tensor(code, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        lerped_sample = model.p_sample_loop(shape=img.shape, cond=lerped_code, progress=True)
        lerped_sample = lerped_sample.squeeze().detach().cpu().numpy()
        print(lerped_sample.shape, lerped_sample.min(), lerped_sample.max())
        all_samples.append(lerped_sample)

      fig, axs = plt.subplots(2, 5, figsize=(10, 5))
      axs[0][0].imshow(original, cmap='hot', vmin=0.0, vmax=1.0)
      axs[0][1].imshow(original_sample, cmap='hot', vmin=0.0, vmax=1.0)
      axs[0][0].axis("off")
      axs[0][1].axis("off")   
      for ax, d, sample in zip(axs.ravel()[2:], distances, all_samples):
        ax.imshow(sample, cmap='hot', vmin=0.0, vmax=1.0)
        ax.set_title("Distance: {:.2f}".format(d[0]))
        ax.axis("off")
      axs[0][0].set_title("Original (Score: {})".format(score))
      axs[0][1].set_title("Sample")
      plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.05, hspace=0.05)
      fig.savefig(f"./lerp-results/examples/image_{i}.pdf", dpi=1200, bbox_inches='tight')
      plt.close(fig)
      counter += 1





if __name__ == "__main__":
    main()