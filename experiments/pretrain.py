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
from datasets import TarFLCDataset
from loaders import get_STED_dataset, get_CTC_dataset, get_JUMP_dataset
from collections import defaultdict
from collections.abc import Mapping
from multiprocessing import Manager
import os
import numpy as np
from tqdm import tqdm
import utils
from model_builder import get_base_model
from torchinfo import summary

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--restore-from", type=str, default="")
parser.add_argument("--model", type=str, default="mae-base")
parser.add_argument("--dataset", type=str, default="STED")
parser.add_argument("--save-folder", type=str, default="./Datasets/FLCDataset/baselines/mae-base_STED")
parser.add_argument("--dataset-path", type=str, default="./Datasets/FLCDataset/dataset.tar")
parser.add_argument("--use-tensorboard", action="store_true")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--modality", type=str, default="STED", choices=["STED", "CTC", "JUMP"])
args = parser.parse_args()

def track_loss(loss):
    fig = plt.figure()
    x = np.arange(0, len(loss), 1)
    plt.plot(x, loss, color='steelblue', label='Train')
    plt.legend()
    fig.savefig(f"{args.save_folder}/{args.modality}_{args.model}_loss.png")
    plt.close(fig)

def load_pretraining_dataset():
    if args.modality == "STED":
        dataloader = get_STED_dataset(transform=None, path=args.dataset_path)
    elif args.modality == "CTC":
        dataloader = ctc_loader(transform=None, path=args.dataset_path)
    elif args.modality == "JUMP":
        dataloader = get_JUMP_dataset(h5file=args.dataset_path, n_channels=1, transform=None)
    else:
        raise NotImplementedError(f"{args.modality} pretraining dataset not supported.")
    print(f"--- Loaded {args.modality} pretraining dataset ({len(dataloader.dataset)} images)---")
    return dataloader




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

    model, _ = get_base_model(name=args.model)

    # Restore model
    ckpt = checkpoint.get("model", None)
    if not ckpt is None:
        print("--- Restoring model ---")
        model.load_state_dict(ckpt)
    model = model.to(DEVICE)

    summary(model)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        MAETransform(input_size=224),
    ])
   

    dataloader = load_pretraining_dataset()
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

    epochs = list(range(len(stats['mean']), 1600))
    print(len(epochs))
    pbar = tqdm(epochs, leave=False, desc="Training...")
    for epoch in pbar:
        print(f"--- Epoch {epoch} ---")
        stats_loss, running_loss, running_batches = [], 0, 0
        for images in tqdm(dataloader, desc='Images...'):
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
        print(f"Saved model to {os.path.join(OUTPUT_FOLDER, 'current_model.pth')}")
        if epoch % 10 == 0:
            torch.save({
                "optimizer": optimizer.state_dict(),
                "model": model.state_dict(),
                "stats": stats
            }, os.path.join(OUTPUT_FOLDER, f'checkpoint-{epoch}.pth'))
        print(f"Saved model to {os.path.join(OUTPUT_FOLDER, f'checkpoint-{epoch}.pth')}")


if __name__=="__main__":
    main()