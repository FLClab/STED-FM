import numpy as np
import torch
from utils.data_utils import tar_dataloader
from utils.training_utils import SaveBestModel, AverageMeter, track_loss
from experiments.models.mae import MAE
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List
from tqdm import tqdm 
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-path", type=str, default="./Datasets/FLCDataset/dataset.tar")
parser.add_argument("--seed", type=int, default=32)
parser.add_argument("--restore-from", type=str, default="")
parser.add_argument("--save-folder", type=str, default="./Datasets/FLCDataset/baselines/not_lightly")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

Net = torch.nn.Module
Loader = torch.utils.data.DataLoader

def run_single_images(model: MAE, imgs: torch.Tensor, preds: torch.Tensor, masks: torch.Tensor, device: torch.device, num_samples: int = 10, outpath: str = "") -> None:
    for i in range(num_samples):
        x = imgs[i].unsqueeze(dim=0)
        pred = preds[i].unsqueeze(dim=0)
        pred = model.unpatchify(pred)
        pred = torch.einsum('nchw->nhwc', pred).detach().cpu()
        mask = masks[i].detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2*1) # (N, H*W, p*p)
        mask = model.unpatchify(mask)
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
        x = torch.einsum('nchw->nhwc', x).cpu()
        # masked image
        im_masked = x * (1 - mask)

        # MAE reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + pred * mask

        # make the plt figure larger
        fig, axs = plt.subplots(1, 4)
        axs[0].imshow(x[0], cmap='hot')
        axs[1].imshow(im_masked[0], cmap='hot')
        axs[2].imshow(pred[0], cmap='hot')
        axs[3].imshow(im_paste[0], cmap='hot')
        for t, ax in zip(["Original", "Masked", "Reconstruction", "Reconstruction+\nvisible"], axs):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(t)
        fig.savefig(f"{outpath}/sanity_check/mae_example_{i}.png")
        plt.close(fig)
        
def train_one_epoch(dataloader: Loader, model: MAE, optimizer: torch.optim, epoch: int, device: torch.device, outpath: str):
    model.train()
    loss_meter = AverageMeter()
    for batch in tqdm(dataloader, desc="Training..."):
        imgs = batch.to(device)
        loss, preds, masks, _ = model(imgs, mask_ratio=0.75)
        run_single_images(model=model, imgs=imgs, preds=preds, masks=masks, device=device, outpath=outpath)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
    print("Epoch {}: Training loss = {:.3f}({:.3f})".format(epoch + 1, loss_meter.val, loss_meter.avg))
    return loss_meter.avg
    

def train(model: Net, train_loader: Loader, optimizer: torch.optim, num_epochs: int, device: torch.device, outpath: str, scheduler: torch.optim) -> None:
    train_loss, lrates = [], []
    for epoch in range(num_epochs):
        loss = train_one_epoch(dataloader=train_loader, model=model, optimizer=optimizer, epoch=epoch, device=device, outpath=outpath)
        train_loss.append(loss)
        scheduler.step()
        temp_lr = optimizer.param_groups[0]['lr']
        lrates.append(temp_lr)
        track_loss(train_loss=train_loss, lrates=lrates, save_dir=outpath)
        torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': None,
            }, f'{outpath}/current_model.pth')
        if epoch + 1 % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': None,
            }, f'{outpath}/model_{epoch+1}.pth')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 1000 

    # Restore model if necessary and get output path
    if args.restore_from:
        checkpoint = torch.load(args.restore_from)
        OUTPUT_FOLDER = os.path.dirname(args.restore_from)
    else:
        checkpoint = {}
        OUTPUT_FOLDER = args.save_folder
    if args.dry_run:
        OUTPUT_FOLDER = os.path.join(args.save_folder, "debug")


    dataloader = tar_dataloader(path=args.dataset_path, transform=None)
    model = MAE(
        img_size=224,
        patch_size=16, 
        input_channels=1,
        embed_dim=512, 
        depth=12, 
        decoder_embed_dim=256,
        num_heads=8,
        decoder_depth=1,
        decoder_num_heads=8,
        mlp_ratio=4.0
    )
    ckpt = checkpoint.get("model", None)
    if not ckpt is None:
        print("--- Restoring model ---")
        model.load_state_dict(ckpt)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    ckpt = checkpoint.get("optimizer", None)
    if not ckpt is None:
        print(f"--- Restoring optimizer ---")
        optimizer.load_state_dict

    train(
        model=model,
        train_loader=dataloader, 
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        scheduler=scheduler,
        outpath=OUTPUT_FOLDER
    )


if __name__=="__main__":
    main()
