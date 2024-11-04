import numpy as np
import matplotlib.pyplot as plt 
import torch 
from diffusion_models.diffusion.ddpm_lightning import DDPM
from diffusion_models.diffusion.denoising.unet import UNet 
from tqdm import trange, tqdm
from torch import nn 
import os 
import argparse 
import sys 
from datamodule import MultiprocessingDataModule 
from class_dict import class_dict 
sys.path.insert(0, "../")
from model_builder import get_pretrained_model_v2 
from datasets import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset-path", type=str, default="./Datasets/FLCDataset/baselines/dataset.tar")
parser.add_argument("--model", type=str, default="mae-lightning-tiny")
parser.add_argument("--weights", type=str, default="MAE_TINY_STED")
parser.add_argument("--timesteps", type=int, default=1000)
parser.add_argument("--dataset", type=str, default="STED")
parser.add_argument("--checkpoint", type=str, default='/home/frbea320/scratch/model_checkpoints/DiffusionModels/latent-guidance')
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--num-samples", type=int, default=10)
parser.add_argument("--sampling", type=str, default="ddpm")
args = parser.parse_args()

def get_save_folder() -> str: 
    if args.weights is None:
        return "from-scratch"
    elif "imagenet" in args.weights.lower():
        return "ImageNet"
    elif "sted" in args.weights.lower():
        return "STED"
    elif "jump" in args.weights.lower():
        return "JUMP"
    elif "ctc" in args.weights.lower():
        return "CTC"
    elif "hpa" in args.weights.lower():
        return "HPA"
    else:
        raise NotImplementedError("The requested weights do not exist.")

if __name__=="__main__":
    SAVEFOLDER = get_save_folder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_encoder, model_config = get_pretrained_model_v2(
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
    checkpoint = torch.load(args.checkpoint)
    print(list(checkpoint.keys()))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    dataset = get_dataset(
        name=args.dataset,
        path=args.dataset_path,
        use_cache=False,
    )
    N = len(dataset)

    for i in range(5):
        idx = np.random.randint(N)
        img = dataset[idx]
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device) # B = 1
        print(img.shape)
        original = img.squeeze().detach().cpu().numpy()
        fig, axs = plt.subplots(1, 11)
        axs[0].imshow(original, cmap='hot', vmin=0.0, vmax=1.0)
        axs[0].set_title("Original")
        for s in range(args.num_samples):
            condition = model.latent_encoder.forward_features(img)

            if args.sampling == "ddpm":
                sample = model.p_sample_loop(shape=img.shape, cond=condition, progress=True)
            else:
                sample = model.ddim_sample_loop(shape=img.shape, cond=condition, progress=True)
            
            sample = sample.squeeze().detach().cpu().numpy()
            m, M = sample.min(), sample.max()
            sample = (sample - m) / (M - m)
            axs[s + 1].imshow(sample, cmap='hot', vmin=0.0, vmax=1.0)
        for ax in axs:
            ax.axis("off")
        fig.savefig(f"./viz/{SAVEFOLDER}/{args.sampling}/img_{i}.pdf", dpi=1200, bbox_inches='tight')
        plt.close(fig)
            
        