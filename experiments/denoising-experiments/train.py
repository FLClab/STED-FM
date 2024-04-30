import torch
import matplotlib.pyplot as plt 
import numpy as np
import argparse 
from tqdm import tqdm 
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.optim
from datasets import get_dataset
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import os 
import sys
sys.path.insert(0, "../")
from model_builder import get_pretrained_model_v2 
from utils import SaveBestModel, AverageMeter
sys.path.insert(1, '../segmentation-experiments')
from decoders.vit import ViTDecoder

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='synaptic-denoising')
parser.add_argument("--backbone", type=str, default='mae-small')
parser.add_argument("--weights", type=str, default="MAE_SSL_STED")
parser.add_argument("--global-pool", type=str, default='avg')
parser.add_argument("--blocks", type=str, default='all') # freeze entire backbone by default
args = parser.parse_args()

def track_loss(train_loss: list, val_loss: list, lrates: list, save_dir: str) -> None:
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    x = np.arange(0, len(train_loss), 1)
    ax2 = ax1.twinx()
    ax1.plot(x, train_loss, color='steelblue', label="Train")
    ax1.plot(x, val_loss, color='firebrick', label="Validation")
    ax2.plot(x, lrates, color='palegreen', label='lr', ls='--')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    fig.savefig(f"{save_dir}")
    plt.close(fig)

def display_predictions(imgs: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor):
    SAVE_NAME = get_save_folder()
    counter = 0
    for img, mask, pred in zip(imgs, labels, predictions):
        if counter > 20:
            break
        else:
            img = torch.permute(img, dims=(1,2,0)).cpu().detach().numpy()
            mask = torch.permute(mask, dims=(1,2,0)).cpu().detach().numpy()
            pred = torch.permute(pred, dims=(1,2,0)).cpu().detach().numpy()
            fig, axs = plt.subplots(1 , 3)
            axs[0].imshow(img, cmap='hot')
            axs[1].imshow(mask, cmap='hot')
            axs[2].imshow(pred, cmap='hot')
            for ax, t in zip(axs, ["Confocal", "STED", "Prediction"]):
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(t)
            fig.savefig(f"./results/mae-small_{SAVE_NAME}/example_{counter}.png", dpi=1200, bbox_inches='tight')
            plt.close(fig)
            counter += 1

def denoising_loss(predictions: torch.Tensor, labels: torch.Tensor, sim_weight: float = 1.0):
    recon_loss = torch.nn.functional.mse_loss(predictions, labels)
    sim_loss = 1 - ms_ssim(predictions, labels, data_range=1.0, size_average=True)
    return recon_loss + (sim_weight * sim_loss)
        

def validation_step(model, valid_loader, epoch, device):
    model.eval()
    loss_meter = AverageMeter()
    with torch.no_grad():
        for imgs, labels in tqdm(valid_loader, desc="Validation..."):
            imgs, labels = imgs.to(device), labels.to(device)
            predictions = model(imgs)
            loss = denoising_loss(predictions, labels)
            loss_meter.update(loss.item())
    display_predictions(imgs=imgs, labels=labels, predictions=predictions)
    return loss_meter.avg

def train_one_epoch(train_loader, model, optimizer, epoch, device):
    model.train()
    loss_meter = AverageMeter()
    for imgs, labels in tqdm(train_loader, desc="Training..."):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(imgs)
        loss = denoising_loss(predictions, labels)
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
    print("Epoch {} loss = {:.3f} ({:.3f})".format(
        epoch + 1, loss_meter.val, loss_meter.avg))
    return loss_meter.avg


def train(
    model,
    train_loader,
    valid_loader,
    device,
    num_epochs,
    optimizer,
    scheduler,
    model_path
):
    train_loss, val_loss, lrates = [], [], []
    save_best_model = SaveBestModel(save_dir=model_path, model_name=f"frozen_{args.blocks}blocks_segmentation_model")
    for epoch in tqdm(range(num_epochs), desc="Epochs..."):
        loss = train_one_epoch(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer, 
            epoch=epoch,
            device=device
        )
        train_loss.append(loss)
        v_loss= validation_step(model=model, valid_loader=valid_loader, epoch=epoch, device=device)
        val_loss.append(v_loss)
        scheduler.step()
        temp_lr = optimizer.param_groups[0]['lr']
        lrates.append(temp_lr)
        save_best_model(v_loss, epoch=epoch, model=model, optimizer=optimizer, criterion=None)
        track_loss(
            train_loss, 
            val_loss, 
            lrates, 
            save_dir=f"{model_path}/frozen_{args.blocks}blocks_denoising_curves.png")
        torch.save({
            "optimizer": optimizer.state_dict(),
            "model": model.state_dict(),
            "epoch": epoch + 1
        }, os.path.join(model_path, f"frozen_{args.blocks}blocks_denoising_epoch{epoch+1}_model.pth"))

def get_save_folder() -> str:
    if "imagenet" in args.weights.lower():
        return "ImageNet"
    elif "sted" in args.weights.lower():
        return "STED"
    else:
        return "CTC"


def main():
    SAVE_NAME = get_save_folder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_channels = 3 if "imagenet" in args.weights.lower() else 1 

    backbone, cfg = get_pretrained_model_v2(
        name=args.backbone, 
        weights=args.weights,
        as_classifier=False,
        blocks='all',
        path=None,
        mask_ratio=0.0, 
        pretrained=True if "imagenet" in args.weights.lower() else 1,
        in_channels=n_channels,
        )
    backbone = backbone.backbone # We get the ViT encoder part of the LightlyMae

    cfg.batch_size = 32

    print(f"--- Loaded backbone ---")
    train_dataset, val_dataset, _ = get_dataset(
        name=args.dataset, cfg=cfg, n_channels=n_channels)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=6
    )
    valid_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=6,
    )
    cfg.backbone = "mae"
    model = ViTDecoder(backbone=backbone, cfg=cfg, in_channels=n_channels, out_channels=1, extract_layers=[3, 6, 9, 12])
    model = model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # criterion = torch.nn.MSELoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=100)

    train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        num_epochs=100,
        optimizer=optimizer,
        scheduler=scheduler,
        model_path=f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/denoising-baselines/mae_{SAVE_NAME}/{args.backbone}/{args.dataset}"
    )


if __name__=="__main__":
    main()
