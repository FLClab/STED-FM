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
from attribute_datasets import get_dataset 
import sys
sys.path.insert(0, "../")
from DEFAULTS import BASE_PATH 
from model_builder import get_pretrained_model_v2

parser = argparse.ArgumentParser()
parser.add_argument("--latent-encoder", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--timesteps", type=int, default=1000)
parser.add_argument("--boundary", type=str, default="quality")
parser.add_argument("--num-samples", type=int, default=10)
parser.add_argument("--ckpt-path", type=str, default="/home/frbea320/scratch/model_checkpoints/DiffusionModels/latent-guidance")
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


def load_boundary(boundary: str) -> np.ndarray:
    return np.load(f"./lerp-results/boundaries/{boundary}/optim_{boundary}_boundary.npz")["boundary"]

def load_quality_net() -> nn.Module:
    quality_net = NetTrueFCN()
    quality_checkpoint = torch.load(f"./QualityNet/trained_models/actin/params.net")
    quality_net.load_state_dict(quality_checkpoint)
    return quality_net

def infer_quality(img: torch.Tensor, mask: torch.Tensor, quality_net: nn.Module) -> float:
    quality_net.eval()
    with torch.no_grad():
        y, score = quality_net(img, mask)
    print(y.shape, score.shape)
    return score.item()

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    boundary = load_boundary(args.boundary)
    quality_net = load_quality_net()
    quality_net.to(DEVICE)

    latent_encoder, model_config = get_pretrained_model_v2(
        name=args.latent_encoder,
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

    dataset = get_dataset(name=args.boundary, classes=["actin"])
    N = len(dataset)
    indices = np.arange(N)
    print(f"Dataset size: {N}")
    np.random.shuffle(indices)
    counter = 0 
    for i in tqdm(indices):
        if counter > args.num_samples:
            exit()
        img, metadata = dataset[i]
        score = metadata["score"]
        original = metadata["original"]
        if args.boundary == "quality" and score > 0.50:
            continue 
        original = torch.tensor(original, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        mask = torch.tensor(metadata["mask"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        pred = infer_quality(original, mask, quality_net)
        print(f"Score: {score}, Pred: {pred}")
        fig = plt.figure(figsize=(5,5,))
        plt.imshow(original.cpu().numpy().squeeze(), cmap="hot")
        plt.title(f"Score: {score}, Pred: {pred}")
        plt.xticks([])
        plt.yticks([])
        fig.savefig(f"./QualityNet/temp{i}.png", bbox_inches="tight", dpi=1200)
        plt.close()
        counter += 1 


if __name__=="__main__":
    main()

