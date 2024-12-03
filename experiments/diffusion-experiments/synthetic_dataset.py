import numpy as np 
import matplotlib.pyplot as plt 
import argparse 
from diffusion_models.diffusion.ddpm_lightning import DDPM 
from diffusion_models.diffusion.denoising.unet import UNet 
import torch 
from tqdm import trange, tqdm 
import copy 
import sys 
import tarfile
import random 
sys.path.insert(0, "../")
from datasets import get_dataset 
from model_builder import get_pretrained_model_v2 

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset-path", type=str, default="/home/frbea320/projects/def-flavielc/datasets/FLCDataset/dataset.tar")
parser.add_argument("--latent-encoder", type=str, default="mae-lightning-tiny")
parser.add_argument("--weights", type=str, default="MAE_TINY_STED")
parser.add_argument("--ckpt-path", type=str, default="/home/frbea320/scratch/model_checkpoints/DiffusionModels/latent-guidance")
parser.add_argument("--save-folder", type=str, default="/home/frbea320/projects/def-flavielc/datasets/FLCDataset/")
parser.add_argument("--num-replica", type=int, default=8)
args = parser.parse_args()

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {DEVICE} ---")

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
    model = DDPM(
        denoising_model=denoising_model,
        timesteps=1000,
        beta_schedule="linear",
        condition_type="latent",
        latent_encoder=latent_encoder,
    )

    ckpt = torch.load(f"{args.ckpt_path}/STED/checkpoint-69.pth")
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(DEVICE)

    dataset = get_dataset(
        name="STED",
        path=args.dataset_path,
        use_cache=True, 
        return_metadata=True
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=10, drop_last=False)
    with torch.no_grad():
        for imgs, metadata in tqdm(dataloader):
            condition = latent_encoder.forward_features(imgs)
            image_ids = metadata["image-id"]
            for sample_id in range(args.num_replica):
                print(image_ids)
                exit()




if __name__=="__main__":
    main()