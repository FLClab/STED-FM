import torch
import torchvision
from timm.models.vision_transformer import vit_small_patch16_224
import lightly.models.utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from lightly.transforms import MAETransform
import argparse
import random
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
import typing 
from data.datasets import TarFLCDataset
from utils.data_utils import tar_dataloader
from collections import defaultdict
from collections.abc import Mapping
from multiprocessing import Manager
import os
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--restore-from", type=str, default="")
parser.add_argument("--save-folder", type=str, default="./Datasets/FLCDataset/baselines")
parser.add_argument("--dataset-path", type=str, default="./Datasets/FLCDataset/dataset.tar")
parser.add_argument("--use-tensorboard", action="store_true")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

class LightlyMAE(torch.nn.Module):
    def __init__(self, vit) -> None:
        super().__init__()
        decoder_dim = 512
        self.mask_ratio = 0.75
        self.patch_size = vit.patch_embed.patch_size[0]
        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length
        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=decoder_dim,
            in_chans=1,
            decoder_depth=1,
            decoder_num_heads=8,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
        )

    def forward_encoder(self, images: torch.Tensor, idx_keep: bool=None) -> torch.Tensor:
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x: torch.Tensor, idx_keep: bool, idx_mask: bool) -> torch.Tensor:
        batch_size = x.shape[0]
        x_decode = self.decoder.embed(x)
        x_masked = lightly.models.utils.repeat_token(self.decoder.mask_token, (batch_size, self.sequence_length))
        x_masked = lightly.models.utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))
        x_decoded = self.decoder.decode(x_masked)
        x_pred = lightly.models.utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        idx_keep, idx_mask = lightly.models.utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device
        )
        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(x=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask)
        patches = lightly.models.utils.patchify(images, self.patch_size)
        target = lightly.models.utils.get_at_index(patches, idx_mask-1)
        return x_pred, target

def set_seeds():
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

def reconstruction_examples():
    pass

def track_loss(loss):
    fig = plt.figure()
    x = np.arange(0, len(loss), 1)
    plt.plot(x, loss, color='steelblue', label='Train')
    plt.legend()
    fig.savefig("./MAE_loss.png")
    plt.close(fig)


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {DEVICE} ---")
    if args.restore_from:
        checkpoint = torch.load(args.restore_from)
        OUTPUT_FOLDER = os.path.dirname(args.restore_from)
    else:
        checkpoint = {}
        OUTPUT_FOLDER = args.save_folder
    if args.dry_run:
        OUTPUT_FOLDER = os.path.join(args.save_folder, "debug")

    # if args.use_tensorboard:
    #     writer = SummaryWriter(os.path.join(OUTPUT_FOLDER, "logs"))

    vit = vit_small_patch16_224(in_chans=1)
    model = LightlyMAE(vit)

    # Restore model
    ckpt = checkpoint.get("model", None)
    if not ckpt is None:
        print("--- Restoring model ---")
        model.load_state_dict(ckpt)
    model = model.to(DEVICE)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        MAETransform(input_size=224),
    ])
    transform = None
    dataloader = tar_dataloader(transform=transform)

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # Restore optimizer
    ckpt = checkpoint.get("optimizer", None)
    if not ckpt is None:
        print("--- Restoring optimizer ---")
        optimizer.load_state_dict(ckpt)


    # Restore stats
    stats = defaultdict(list)
    ckpt = checkpoint.get("stats", None)
    if not ckpt is None:
        stats = ckpt

    epochs = list(range(len(stats['mean']), 1000))
    print(len(epochs))
    pbar = tqdm(epochs, leave=False, desc="Training...")
    for epoch in pbar:
        print(f"--- Epoch {epoch} ---")
        stats_loss, running_loss, running_batches = [], 0, 0
        for images in dataloader:
            images = images.to(DEVICE)
            # images = views[0].to(DEVICE)
            predictions, targets = model(images)
            loss = criterion(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_batches += 1
            running_loss += loss.item()
            pbar.set_description(f"--- loss: {running_loss/running_batches:0.4f}")
            stats_loss.append(loss.item())
        for key, func in zip(["mean", "std", "min", "max", "median"], [np.mean, np.std, np.min, np.max, np.median]):
            stats[key].append(func(stats_loss))
        track_loss(loss=stats['mean'])
        scheduler.step()

        torch.save({
            "optimizer": optimizer.state_dict(),
            "model": model.state_dict(),
            "stats": stats
        }, os.path.join(OUTPUT_FOLDER, "current_model.pth"))
        if epoch % 10 == 0:
            torch.save({
                "optimizer": optimizer.state_dict(),
                "model": model.state_dict(),
                "stats": stats
            }, os.path.join(OUTPUT_FOLDER, f"checkpoint-{epoch}.pth"))


if __name__=="__main__":
    main()
