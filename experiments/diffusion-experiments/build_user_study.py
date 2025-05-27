import numpy as np 
import matplotlib.pyplot as plt 
import argparse 
from diffusion_models.diffusion.ddpm_lightning import DDPM
from diffusion_models.diffusion.denoising.unet import UNet 
import torch
import io
from tqdm import trange, tqdm 
import copy
import sys 
import tarfile
from class_dict import class_dict
from stedfm.model_builder import get_pretrained_model_v2
from stedfm.datasets import get_dataset 

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=str, default="/home/frbea320/scratch/model_checkpoints/DiffusionModels/latent-guidance")
parser.add_argument("--latent-encoder", type=str, default="mae-lightning-tiny")
parser.add_argument("--timesteps", type=int, default=1000)
parser.add_argument("--dataset-path", default="")
parser.add_argument("--save-folder", type=str, default="/home/frbea320/scratch/Datasets/FLCDataset/diffusion-user-study")
parser.add_argument("--num-classes", type=int, default=24)
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

if __name__=="__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_FOLDER = args.save_folder 
    MODELS = {
        "MAE_TINY_IMAGENET1K_V1": None,
        "MAE_TINY_JUMP": None,
        "MAE_TINY_HPA": None,
        "MAE_TINY_STED": None,
    }
    for key in MODELS.keys():
        channels = 3 if key == "MAE_TINY_IMAGENET1K_V1" else 1
        latent_encoder, model_config = get_pretrained_model_v2(
            name=args.latent_encoder,
            weights=key,
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
            channels=channels,
            dim_mults=(1,2,4),
            cond_dim=model_config.dim,
            condition_type="latent",
            num_classes=4
        )
        model = DDPM(
            denoising_model=denoising_model,
            timesteps=args.timesteps,
            beta_schedule="linear",
            condition_type="latent",
            latent_encoder=latent_encoder,
        )

        model_path = get_save_folder(key)
        ckpt = torch.load(f"{args.checkpoint}/{model_path}/checkpoint-69.pth")
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(DEVICE)
        MODELS[key] = model

    dataset = get_dataset(
        name="STED",
        path=args.dataset_path,
        use_cache=False,
        return_metadata=True
    )
    N = len(dataset)
    counters = {
        6: 0, # f-actin
        14: 0, # psd-95
        18: 0, # tom20,
        19: 0, # tubulin
    }
    lut = {
        6: "f-actin",
        14: "psd95",
        18: "tom20",
        19: "tubulin"
    }
    print(f"Dataset length: {len(dataset)}")
    for idx in range(N):
        original_img, metadata = dataset[idx]
        protein_id_str = metadata["protein-id"]
        protein_id = class_dict[protein_id_str]
        print(f"{protein_id_str} -> {protein_id}")
        if protein_id not in [6, 14, 18, 19]:
            continue 
        if counters[protein_id] >= 5: # We take 5 originals from each class (20 originals). 4 generations for each -> 80 samples
            continue 
        if sum(list(counters.values())) >= 20:
            print(f"Finished; sampled {counters}")
            break
        else:
            counters[protein_id] += 1
    
        img = torch.tensor(original_img, dtype=torch.float32).unsqueeze(0).to(DEVICE) 

          
        condition = model.latent_encoder.forward_features(img)

        SAMPLES = {}
        for model_key, model in MODELS.items():
            sample = model.p_sample_loop(shape=img.shape, cond=condition, progress=True)
            sample = sample.squeeze().detach().cpu().numpy()
            m, M = sample.min(), sample.max() 
            sample = (sample - m) / (M - m)
            SAMPLES[model_key] = sample

        with tarfile.open(f"{args.save_folder}/mae-small-diffusion-classification-dataset.tar", "a") as handle:
            imagenet = SAMPLES["MAE_TINY_IMAGENET1K_V1"]
            jump = SAMPLES["MAE_TINY_JUMP"]
            hpa = SAMPLES["MAE_TINY_HPA"]
            sted = SAMPLES["MAE_TINY_STED"]
            buffer = io.BytesIO()
            np.savez(
                file=buffer,
                image=original_img,
                imagenet=imagenet,
                jump=jump,
                hpa=hpa,
                sted=sted,
                protein_id=protein_id
            )
            buffer.seek(0)
            tarinfo = tarfile.info(name=f"{lut[protein_id]}_{counters[protein_id]}") 
            tarinfo.size = len(buffer.getbuffer())
            handle.addfile(tarinfo=tarinfo, fileobj=buffer)


                

        
