import numpy as np 
import matplotlib.pyplot as plt 
import argparse 
from diffusion_models.diffusion.ddpm_lightning import DDPM 
from diffusion_models.diffusion.denoising.unet import UNet 
import torch 
from tqdm import trange, tqdm  
import copy 
import sys 
import os
import random
from class_dict import class_dict
from attribute_datasets import OptimQualityDataset
from stedfm.DEFAULTS import BASE_PATH 
from stedfm.datasets import get_dataset 
from stedfm.model_builder import get_pretrained_model_v2

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=32)
parser.add_argument("--dataset-path", type=str, default="/home/frbea320/projects/def-flavielc/datasets/FLCDataset/dataset-250k.tar")
parser.add_argument("--latent-encoder", type=str, default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--ckpt-path", type=str, default="/home/frbea320/scratch/model_checkpoints/DiffusionModels/latent-guidance")
parser.add_argument("--num-samples", type=int, default=60)
parser.add_argument("--guidance", type=str, default="latent")
args = parser.parse_args()

def get_save_folder(key: str) -> str: 
    if key is None:
        return "from-scratch"
    elif "imagenet" in key.lower():
        return "ImageNet"
    elif "sted" in key.lower():
        return "STED"
    elif "jump" in key.lower():
        return "JUMP"
    elif "sim" in key.lower():
        return "SIM"
    elif "hpa" in key.lower():
        return "HPA"
    elif "sim" in key.lower():
        return "SIM"
    else:
        raise NotImplementedError("The requested weights do not exist.")

def save_image(image: np.ndarray, generation: np.ndarray, i: int, class_name: str) -> None:
    fig = plt.figure()
    plt.imshow(image, cmap='hot', vmin=0, vmax=1)
    plt.axis("off")
    plt.savefig(f"./classification-study/{args.guidance}-guidance/templates/template{i}_{class_name}.png", dpi=1200, bbox_inches="tight")
    plt.close(fig)


    weights = "classifier-guidance" if args.guidance == "class" else args.weights
    fig = plt.figure()
    plt.imshow(generation, cmap='hot', vmin=0, vmax=1)
    plt.axis("off")
    plt.savefig(f"./classification-study/{args.guidance}-guidance/candidates/{weights}_template{i}_{class_name}.png", dpi=1200, bbox_inches="tight")
    plt.close(fig)

def main():
    np.random.seed(args.seed)
    SAVENAME = get_save_folder(key=args.weights)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {DEVICE} ---")
    n_channels = 3 if SAVENAME == "ImageNet" else 1 
    

    latent_encoder, model_config = get_pretrained_model_v2(
        name=args.latent_encoder,
        weights=args.weights,
        path=None, 
        mask_ratio=0.0,
        pretrained=True if n_channels == 3 else False,
        in_channels=n_channels,
        as_classifier=True,
        blocks="all",
        num_classes=4
    )

    denoising_model = UNet(
        dim=64,
        channels=1,
        dim_mults=(1,2,4),
        cond_dim=model_config.dim,
        condition_type=args.guidance,
        num_classes=24 if args.guidance == "class" else 4
    )

    model = DDPM(
        denoising_model=denoising_model,
        timesteps=1000,
        beta_schedule="linear",
        condition_type=args.guidance,
        latent_encoder=latent_encoder if args.guidance == "latent" else None,
    )

    path = f"{args.ckpt_path}/{args.weights}/checkpoint-69.pth" if args.guidance == "latent" else f"{args.ckpt_path}/checkpoint-69.pth"
    print(path)
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(DEVICE)
    
    dataset = OptimQualityDataset(
            data_folder="/home-local/Frederic/evaluation-data/optim-data",
            num_samples={"actin": None, "tubulin": None, "CaMKII_Neuron": None, "PSD95_Neuron": None},
            high_score_threshold=0.70,
            low_score_threshold=0.0,
            n_channels=1
        )

    os.makedirs(f"./classification-study/{args.guidance}-guidance/templates", exist_ok=True)
    os.makedirs(f"./classification-study/{args.guidance}-guidance/candidates", exist_ok=True)

    counters = {
        "f-actin": 0,
        "psd95": 0,
        "beta-camkii": 0,
        "tubulin": 0,
    }
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    
    model.eval()
    with torch.no_grad():
        for idx in tqdm(indices, total=len(indices), desc="Processing samples"):
            original_img, metadata = dataset[idx]
            # class_name = metadata["protein-id"]
            protein = metadata["protein"]
        
            if protein == "actin":
                class_name = "f-actin"
            elif protein == "tubulin":
                class_name = "tubulin"
            elif protein == "CaMKII_Neuron":
                class_name = "beta-camkii"
            elif protein == "PSD95_Neuron":
                class_name = "psd95"
            if sum(list(counters.values())) >= args.num_samples:
                print(f"Finished; sampled {counters}")
                break
            elif class_name not in list(counters.keys()):
                continue
            elif counters[class_name] >= args.num_samples // len(counters):
                continue
            else:
                counters[class_name] += 1
                print(counters)
            
            if SAVENAME == "ImageNet":
                image = torch.tensor(original_img, dtype=torch.float32).repeat(3, 1, 1).unsqueeze(0).to(DEVICE)
                assert torch.equal(image[0, 0, :, :], image[0, 1, :, :]) and torch.equal(image[0, 1, :, :], image[0, 2, :, :]), "All three channels in the image tensor are not equal"
            else:
                image = torch.tensor(original_img, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            condition = model.latent_encoder.forward_features(image) if args.guidance == "latent" else torch.tensor(class_dict[class_name], dtype=torch.int8).to(DEVICE).long()

            original_img = original_img[0] 
            # m, M = original_img.min(), original_img.max()
            # original_img = (original_img - m) / (M - m)
            
            with torch.no_grad():
                generation = model.p_sample_loop(shape=(image.shape[0], 1, image.shape[2], image.shape[3]), cond=condition, progress=True)
                
            generation = generation.squeeze().cpu().numpy()
            # image = image.squeeze().cpu().numpy()
            
            save_image(original_img, generation, idx, class_name)

if __name__ == "__main__":
    main()
