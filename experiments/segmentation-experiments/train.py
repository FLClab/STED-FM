import torch 
import matplotlib.pyplot as plt
import numpy as np
import argparse  
from tqdm import tqdm 
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim
from torch.utils.data import DataLoader
from decoders.unet import UNet
from decoders.vit import ViTDecoder
from torchinfo import summary
from datasets import get_dataset
import os
import sys 
sys.path.insert(0, "../")
from model_builder import get_base_model, get_pretrained_model_v2
from utils import SaveBestModel, AverageMeter, compute_Nary_accuracy

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='synaptic-segmentation')
parser.add_argument("--backbone", type=str, default='mae-small')
parser.add_argument("--weights", type=str, default="MAE_SSL_STED")
parser.add_argument("--global-pool", type=str, default='avg')
parser.add_argument("--blocks", type=str, default='all') # freeze backbone by default
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

def display_predictions(imgs: torch.Tensor, masks: torch.Tensor, predictions: torch.Tensor):
    counter = 0
    for img, mask, pred in zip(imgs, masks, predictions):
        if counter > 20:
            break
        else:
            img = torch.permute(img, dims=(1,2,0)).cpu().detach().numpy()
            mask = torch.permute(mask, dims=(1,2,0)).cpu().detach().numpy()
            pred = torch.permute(pred, dims=(1,2,0)).cpu().detach().numpy()
            fig, axs = plt.subplots(1 , 3)
            axs[0].imshow(img, cmap='hot')
            axs[1].imshow(mask, cmap='gray')
            axs[2].imshow(pred, cmap='gray')
            for ax, t in zip(axs, ["Image", "Ground truth", "Prediction"]):
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(t)
            fig.savefig(f"./results/mae-small/example_{counter}.png", dpi=1200, bbox_inches='tight')
            plt.close(fig)
            counter += 1
        

def validation_step(model, valid_loader, criterion, epoch, device):
    model.eval()
    loss_meter = AverageMeter()
    with torch.no_grad():
        for imgs, data_dict in tqdm(valid_loader, desc="Validation..."):
            masks = data_dict['label']
            imgs, masks = imgs.to(device), masks.to(device)
            predictions = model(imgs)
            loss = criterion(predictions, masks)
            loss_meter.update(loss.item())
    display_predictions(imgs=imgs, masks=masks, predictions=predictions)
    return loss_meter.avg

def train_one_epoch(train_loader, model, optimizer, criterion, epoch, device):
    model.train()
    loss_meter = AverageMeter()
    for imgs, data_dict in tqdm(train_loader, desc="Training..."):
        labels = data_dict['label']
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(imgs)
        loss = criterion(predictions, labels)
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
    criterion, 
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
            criterion=criterion,
            epoch=epoch,
            device=device
        )
        train_loss.append(loss)
        v_loss= validation_step(model=model, valid_loader=valid_loader, criterion=criterion, epoch=epoch, device=device)
        val_loss.append(v_loss)
        scheduler.step(v_loss)
        temp_lr = optimizer.param_groups[0]['lr']
        lrates.append(temp_lr)
        save_best_model(v_loss, epoch=epoch, model=model, optimizer=optimizer, criterion=criterion)
        track_loss(
            train_loss, 
            val_loss, 
            lrates, 
            save_dir=f"{model_path}/frozen_{args.blocks}blocks_segmentation_curves.png")
        torch.save({
            "optimizer": optimizer.state_dict(),
            "model": model.state_dict(),
            "epoch": epoch + 1
        }, os.path.join(model_path, f"frozen_{args.blocks}blocks_segmentation_epoch{epoch+1}_model"))

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
        name=args.dataset, cfg=cfg)

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
    model = ViTDecoder(backbone=backbone, cfg=cfg, in_channels=1, out_channels=1, extract_layers=[3, 6, 9, 12])
    model = model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, patience=10, threshold=0.01, min_lr=1e-5, factor=0.9)

    train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        num_epochs=100,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        model_path=f"/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/segmentation-baselines/mae_STED/{args.backbone}/{args.dataset}"
    )


if __name__=="__main__":
    main()
