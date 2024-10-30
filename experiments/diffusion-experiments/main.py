import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from models.diffusion.ddpm import DDPM
from models.diffusion.denoising.unet import UNet
from tqdm import trange, tqdm
import argparse 
from torch.utils.data import DataLoader 
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
import torch.distributed as dist
import torch.utils.data.distributed
from class_dict import class_dict
import os
sys.path.insert(0, "../")
from DEFAULTS import BASE_PATH
from datasets import get_dataset
from utils import SaveBestModel, AverageMeter, compute_Nary_accuracy, track_loss, update_cfg, get_number_of_classes 
plt.style.use("dark_background")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset-path", type=str, default="./Datasets/FLCDataset/baselines/dataset.tar")
parser.add_argument("--model", default="mae-lightning-small")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--timesteps", type=int, default=1000)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--num-classes", type=int, default=24)
parser.add_argument("--checkpoint", type=str, default=None)

args = parser.parse_args()

def sample(ddpm, epoch):
    ddpm.eval()
    with torch.no_grad():
        conditions = np.random.randint(args.num_classes, size=20)
        conditions = torch.tensor(conditions, dtype=torch.int8)
        names = list(class_dict.keys())
        samples = ddpm.p_sample_loop(shape=(20, 1, 224, 224), progress=True)
        for i in range(samples.shape[0]):
            img = samples[i].squeeze().detach().cpu().numpy()# .reshape(64, 64, 1)
            m, M = img.min(), img.max()
            img = (img - m) / (M - m)
            fig = plt.figure()
            plt.imshow(img, cmap='hot', vmin=0.0, vmax=1.0)
            plt.title(names[conditions[i]])
            plt.xticks([])
            plt.yticks([])
            fig.savefig(f"./model-checkpoints/classifier-guidance_epoch{epoch}_sample_{i}.png", dpi=1200)
            plt.close(fig)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpy")

    dataset = get_dataset(name="STED", path=args.dataset_path, return_metadata=True)
    dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=6, drop_last=False)
    
    denoising_model = UNet(
        dim=64,
        channels=1,
        dim_mults=(1,2,4),
        condition_type="class",
        num_classes=args.num_classes
    )
    ddpm = DDPM(
        denoising_model=denoising_model,
        timesteps=args.timesteps,
        beta_schedule="linear",
    )
    ddpm = ddpm.to(device)
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=2e-4, betas=(0.9, 0.99))

    start_epoch = 0
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        ddpm.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optmizer_state_dict"])

    train_loss, lrates = [], [] 
    save_best_model = SaveBestModel(save_dir="./model-checkpoints", model_name="classifier_guided_DDPM")
    for epoch in trange(start_epoch, args.epochs, desc="Epochs..."):
        ddpm.train()
        loss_meter = AverageMeter()
        for batch in tqdm(dataloader):
            imgs, metadata = batch
            imgs = imgs.to(device)
            protein_ids = metadata["protein-id"]
            cls_labels = [class_dict[key] for key in protein_ids]
            cls_labels = torch.tensor(cls_labels, dtype=torch.int8).to(device).long()
            optimizer.zero_grad()
            t = torch.randint(0, args.timesteps, (imgs.shape[0],), device=device).long()
            losses, model_outputs = ddpm(x_0=imgs, t=t, cond=cls_labels)
            loss = losses["loss"].mean()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())

        train_loss.append(loss_meter.avg)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(train_loss, label="Train", color="tab:blue")
        ax.legend()
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        fig.savefig(f"./model-checkpoints/{save_best_model.model_name}-training-curves.png")
        plt.close(fig)

        sample(ddpm=ddpm, epoch=epoch)
        
        if epoch % 10 == 0:
            torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': ddpm.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f'./model-checkpoints/{save_best_model.model_name}-checkpoint{epoch+1}.pth')
        

        


if __name__=="__main__":
    main()