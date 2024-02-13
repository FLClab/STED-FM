import numpy as np
import torch
from utils.data_utils import tar_dataloader
from models.mae import MAE, MaskedAutoencoderViT
from typing import List
from tqdm import tqdm 
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--datapath", type=str)
args = parser.parse_args()

Net = torch.nn.Module
Loader = torch.utils.data.DataLoader

def run_single_images(model: MAE, imgs: torch.Tensor, device):
    img = torch.rand(224, 224, 1)
    img = img.unsqueeze(dim=0)
    img = torch.einsum('nhwc->nchw', img).to(device)
    loss, y, mask, _ = model(img.float(), mask_ratio=0.75)
    print(f"In run one image: {y.shape}")
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *1)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    img = torch.einsum('nchw->nhwc', img).cpu()
    # masked image
    im_masked = img * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = img * (1 - mask) + y * mask
    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(img[0], cmap='hot')
    axs[1].imshow(im_masked[0], cmap='hot')
    axs[2].imshow(y[0], cmap='hot')
    axs[3].imshow(im_paste[0], cmap='hot')
    for t, ax in zip(["Original", "Masked", "Reconstruction", "Reconstruction+\nvisible"], axs):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(t)
    fig.savefig(f"./temp.png")
    plt.close(fig)

def train_one_epoch(dataloader: Loader, model: MAE, optimizer: torch.optim, epoch: int, device: torch.device):
    model.train()
    count = 0
    for batch in tqdm(dataloader, desc="Training..."):
        count += 1
        imgs = batch.to(device)
        losses, preds, masks, _ = model(imgs, mask_ratio=0.75)
        print(f"In train one epoch: {preds.shape}")
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if count == 5:
            for i in range(imgs.shape[0]):
                x = imgs[i].unsqueeze(0)
                pred = preds[i].unsqueeze(dim=0)
                print(f"In pred: {pred.shape}")
                pred = model.unpatchify(pred)
                pred = torch.einsum('nchw->nhwc', pred).detach().cpu()

                mask = masks[i].detach()
                mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *1)  # (N, H*W, p*p*3)
                print(f"In mask: {mask.shape}")
                mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
                mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
                print(f"X Shape {x.shape}")
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
                fig.savefig(f"./temp_{count}")
            exit()


def train(model: Net, train_loader: Loader, optimizer: torch.optim, num_epochs: int, device: torch.device) -> None:
    train_loss = []
    for epoch in range(num_epochs):
        loss = train_one_epoch(dataloader=train_loader, model=model, optimizer=optimizer, epoch=epoch, device=device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 1 # For testing
    dataloader = tar_dataloader(path=args.datapath)
    # model = MAE(
    #     img_size=224, 
    #     patch_size=16,
    #     input_channels=1,
    #     embed_dim=512,
    #     depth=12,
    #     decoder_embed_dim=256,
    #     num_heads=8,
    #     decoder_depth=6,
    #     decoder_num_heads=4,
    #     mlp_ratio=4.0
    # ).to(device)
    model = MaskedAutoencoderViT(
        img_size=224,
        patch_size=16, 
        in_chans=1,
        embed_dim=512, 
        depth=12, 
        decoder_embed_dim=256,
        num_heads=8,
        decoder_depth=6,
        decoder_num_heads=4,
        mlp_ratio=4.0
    ).to(device)
    # run_one_image(model=model, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train(
        model=model,
        train_loader=dataloader, 
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device
    )


if __name__=="__main__":
    main()