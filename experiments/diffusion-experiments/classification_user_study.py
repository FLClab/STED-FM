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
import random
sys.path.insert(0, "../")
from DEFAULTS import BASE_PATH 
from datasets import get_dataset 
from model_builder import get_pretrained_model_v2

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset-path", type=str, default="/home/frbea320/projects/def-flavielc/datasets/FLCDataset/dataset-250k.tar")
parser.add_argument("--latent-encoder", type=str, default="mae-lightning-tiny")
parser.add_argument("--weights", type=str, default="MAE_TINY_STED")
parser.add_argument("--ckpt_path", type=str, default="/home/frbea320/scratch/model_checkpoints/DiffusionModels/latent-guidance")
parser.add_argument("--num-samples", type=int, default=40)
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
    elif "ctc" in key.lower():
        return "CTC"
    elif "hpa" in key.lower():
        return "HPA"
    else:
        raise NotImplementedError("The requested weights do not exist.")

def save_image(image: np.ndarray, generation: np.ndarray, i: int, class_name: str) -> None:
    fig = plt.figure()
    plt.imshow(image, cmap='hot')
    plt.axis("off")
    plt.savefig(f"./classification-study/templates/template{i}_{class_name}.png", dpi=1200, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(generation, cmap='hot')
    plt.axis("off")
    plt.savefig(f"./classification-study/candidates/{args.weights}_template{i}_{class_name}.png", dpi=1200, bbox_inches="tight")
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
        channels=n_channels,
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

    ckpt = torch.load(f"{args.ckpt_path}/{SAVENAME}/checkpoint-69.pth")
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(DEVICE)

    dataset = get_dataset(
        name="STED",
        path=args.dataset_path,
        use_cache=False,
        return_metadata=True
    )

    os.makedirs("./classification-study/templates", exist_ok=True)
    os.makedirs("./classification-study/candidates", exist_ok=True)

    # counters = {
    #     "f-actin": 0,
    #     "psd95": 0,
    #     "tom20": 0,
    #     "tubulin": 0
    # }
    counters = {"beta-camkii": 0}
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    
    model.eval()
    with torch.no_grad():
        for idx in tqdm(indices, total=len(indices), desc="Processing samples"):
            original_img, metadata = dataset[idx]
            class_name = metadata["protein-id"]
            if sum(list(counters.values())) >= 10: # args.num_samples:
                print(f"Finished; sampled {counters}")
                break
            elif class_name not in list(counters.keys()):
                continue
            elif counters[class_name] >= 10: # args.num_samples // len(counters):
                continue
            else:
                counters[class_name] += 1
                print(counters)
            
            image = torch.tensor(original_img, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            condition = model.latent_encoder.forward_features(image)

            original_img = original_img[0] 
            m, M = original_img.min(), original_img.max()
            original_img = (original_img - m) / (M - m)
            
            with torch.no_grad():
                generation = model.p_sample_loop(shape=image.shape, cond=condition, progress=True)
                
            generation = generation.squeeze().cpu().numpy()
            image = image.squeeze().cpu().numpy()
            
            save_image(image, generation, idx, class_name)

if __name__ == "__main__":
    main()
