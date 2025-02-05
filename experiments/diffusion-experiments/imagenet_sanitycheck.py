import numpy as np
import matplotlib.pyplot as plt 
import torch 
from diffusion_models.diffusion.ddpm_lightning import DDPM 
from diffusion_models.diffusion.denoising.unet import UNet
import torch 
from tqdm import trange, tqdm 
from torch import nn 
import os 
import argparse 
import attribute_datasets
import sys 
sys.path.insert(0, "../")
from model_builder import get_pretrained_model_v2 
from datasets import get_dataset
import utils 

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset-path", type=str, default="/home-local/Frederic/Datasets/FLCDataset/dataset-250k.tar")
parser.add_argument("--model", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_IMAGENET1K_V1")
parser.add_argument("--timesteps", type=int, default=1000)
parser.add_argument("--dataset", type=str, default="STED")
parser.add_argument("--ckpt-path", type=str, default="/home-local/Frederic/baselines/DiffusionModels/latent-guidance")
parser.add_argument("--num-samples", type=int, default=10)
args = parser.parse_args()


def reconstruct_image(model, img):
    pass 

class DatasetConfig:
    num_workers : int = None
    shuffle : bool = True
    use_cache : bool = True
    max_cache_size : float = 32e+9
    return_metadata : bool = True
    batch_size: int = 4

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = 3 if "imagenet" in args.weights.lower() else 1
    print(f"Using {channels} channels")
    latent_encoder, model_config = get_pretrained_model_v2(
        name=args.model,
        weights=args.weights,
        path=None,
        mask_ratio=0.0,
        pretrained=True if channels == 3 else False,
        in_channels=channels,
        as_classifier=True,
        blocks="all",
        num_classes=4,
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
        latent_encoder=latent_encoder,
    )
    checkpoint = torch.load(f"{args.ckpt_path}/{args.weights}/checkpoint-69.pth")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()

    dataset = get_dataset(
        name=args.dataset,
        path=args.dataset_path,
        use_cache=False,
        return_metadata=True,
        in_channels=channels,
    )
    dataset = attribute_datasets.LowHighResolutionDataset(
        h5path="/home-local/Frederic/evaluation-data/low-high-quality/testing.hdf5",
        n_channels=channels,
    )

    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    indices = np.random.choice(indices, size=args.num_samples, replace=False)
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, metadata = dataset[idx]
            img = img.unsqueeze(0).to(DEVICE)
        
            condition = model.latent_encoder.forward_features(img)
            sample = model.p_sample_loop(shape=(img.shape[0], 1, img.shape[2], img.shape[3]), cond=condition, progress=True)
            sample = sample[:, [0], :, :].squeeze().cpu().detach().numpy()
            img = img[:, [0], :, :].squeeze().cpu().detach().numpy()

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(img, cmap='hot')
            axs[1].imshow(sample, cmap='hot')
            for ax in axs:
                ax.axis('off')
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
            axs[0].set_title("Original")
            axs[1].set_title("Reconstructed")
            fig.savefig(f"./whatsgoingon/resolutiondataset_{i}.png", dpi=1200)
            plt.close(fig)
            print(metadata)
            print("\n")

if __name__ == "__main__":
    main()