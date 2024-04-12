
import torch
import torchvision
import numpy
import os
import typing
import random

from lightly import loss
from lightly import transforms
from lightly.data import LightlyDataset
from lightly.models.modules import heads
from tqdm import tqdm
from collections import defaultdict
from collections.abc import Mapping
from multiprocessing import Manager
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import backbones

from dataset import TarFLCDataset
from modules.transforms import SimCLRTransform
from backbones import get_backbone

# Create a PyTorch module for the SimCLR model.
class SimCLR(torch.nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        self.projection_head = heads.SimCLRProjectionHead(
            input_dim=self.cfg.dim,
            hidden_dim=512,
            output_dim=128,
        )

    def load_state_dict(self, state_dict: Mapping[str, torch.Any], strict: bool = True, assign: bool = False):
        self.backbone.load_state_dict(state_dict["backbone"])
        self.projection_head.load_state_dict(state_dict["projection-head"])
    
    def state_dict(self):
        return {
            "backbone" : self.backbone.state_dict(),
            "projection-head" : self.projection_head.state_dict()
        }

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)
        return z

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")     
    parser.add_argument("--restore-from", type=str, default="",
                    help="Model from which to restore from") 
    parser.add_argument("--save-folder", type=str, default="./data/SSL/baselines",
                    help="Model from which to restore from")     
    parser.add_argument("--dataset-path", type=str, default="./data/FLCDataset/20240214-dataset.tar",
                    help="Model from which to restore from")         
    parser.add_argument("--backbone", type=str, default="resnet18",
                        help="Backbone model to load")
    parser.add_argument("--use-tensorboard", action="store_true",
                        help="Logging using tensorboard")    
    parser.add_argument("--dry-run", action="store_true",
                        help="Activates dryrun")        
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    numpy.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    backbone, cfg = get_backbone(args.backbone)
    
    if args.restore_from:
        checkpoint = torch.load(args.restore_from)
        OUTPUT_FOLDER = os.path.dirname(args.restore_from)
    else:
        checkpoint = {}
        OUTPUT_FOLDER = os.path.join(args.save_folder, args.backbone)
    if args.dry_run:
        OUTPUT_FOLDER = os.path.join(args.save_folder, "debug")
    
    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(OUTPUT_FOLDER, "logs"))

    # Build the SimCLR model.
    model = SimCLR(backbone, cfg)
    ckpt = checkpoint.get("model", None)
    if not ckpt is None:
        print("Restoring model...")
        model.load_state_dict(ckpt)
    model = model.to(DEVICE)

    summary(model, input_size=(1, 224, 224))

    # Prepare transform that creates multiple random views for every image.
    transform = SimCLRTransform(
        input_size=224,
        cj_prob = 0.8,
        cj_strength = 1.0,
        cj_bright = 0.8,
        cj_contrast = 0.8,
        cj_sat = 0,
        cj_hue = 0,
        min_scale = 0.3,
        random_gray_scale = 0,
        gaussian_blur = 0,
        kernel_size = None,
        sigmas = (0.1, 2),
        vf_prob = 0.5,
        hf_prob = 0.5,
        rr_prob = 0.5,
        rr_degrees = None,
        normalize = False,
    )


    # Create a dataset from your image folder.
    manager = Manager()
    cache_system = manager.dict()
    tar_path = args.dataset_path
    if args.dry_run:
        tar_path = "./data/FLCDataset/debug-dataset.tar"
    dataset = TarFLCDataset(
        tar_path=tar_path, transform=transform, 
        use_cache=True, cache_system=cache_system, max_cache_size=16e9)

    # Build a PyTorch dataloader.
    dataloader = torch.utils.data.DataLoader(
        dataset,  # Pass the dataset to the dataloader.
        batch_size=cfg.batch_size,  # A large batch size helps with the learning.
        shuffle=True,  # Shuffling is important!
        num_workers=4
    )

    # Lightly exposes building blocks such as loss functions.
    criterion = loss.NTXentLoss(temperature=0.1)

    # Get a PyTorch optimizer.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-6)
    ckpt = checkpoint.get("optimizer", None)
    if not ckpt is None:
        print("Restoring optimizer...")
        optimizer.load_state_dict(ckpt)

    # Train the model.
    stats = defaultdict(list)
    ckpt = checkpoint.get("stats", None)
    if not ckpt is None:
        stats = ckpt
    for epoch in range(len(stats["mean"]), 1024):
        print(f"[----] Epoch: {epoch}")
        pbar = tqdm(dataloader, leave=False)
        stats_loss, running_loss, running_batches = [], 0, 0
        for batch_idx, (view0, view1) in enumerate(pbar):

            view0 = view0.to(DEVICE)
            view1 = view1.to(DEVICE)

            z0 = model(view0)
            z1 = model(view1)
            loss = criterion(z0, z1)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_batches += 1
            running_loss += loss.item()
            pbar.set_description(f"[----] loss: {running_loss / running_batches:0.4f}")

            stats_loss.append(loss.item())

            if (batch_idx == 0) and args.use_tensorboard:
                writer.add_images("Images/view0", view0[:5], epoch, dataformats="NCHW")
                writer.add_images("Images/view1", view1[:5], epoch, dataformats="NCHW")

        print(f"[----] Epoch: {epoch}")
        for key, func in zip(["mean", "std", "min", "max", "median"], 
                            [numpy.mean, numpy.std, numpy.min, numpy.max, numpy.median]):
            stats[key].append(func(stats_loss))

            if args.use_tensorboard:
                writer.add_scalar(f"Loss/{key}", stats[key][-1], epoch)
        
        torch.save({
            "optimizer" : optimizer.state_dict(),
            "model" : model.state_dict(),
            "stats" : stats
        }, os.path.join(OUTPUT_FOLDER, "result.pt"))
        if epoch % 10 == 0:
            torch.save({
                "optimizer" : optimizer.state_dict(),
                "model" : model.state_dict(),
                "stats" : stats
            }, os.path.join(OUTPUT_FOLDER, f"checkpoint-{epoch}.pt"))