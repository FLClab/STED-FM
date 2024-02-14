
import torch
import torchvision
import numpy

from lightly import loss
from lightly import transforms
from lightly.data import LightlyDataset
from lightly.models.modules import heads
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Manager

from dataset import TarFLCDataset
from transforms import SimCLRTransform

# Create a PyTorch module for the SimCLR model.
class SimCLR(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = heads.SimCLRProjectionHead(
            input_dim=512,  # Resnet18 features have 512 dimensions.
            hidden_dim=512,
            output_dim=128,
        )

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)
        return z

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore-from", type=str, default="",
                        help="Model from which to restore from")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use a resnet backbone.
    backbone = torchvision.models.resnet18()
    backbone.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # Ignore the classification head as we only want the features.
    backbone.fc = torch.nn.Identity()

    if args.restore_from:
        checkpoint = torch.load(args.restore_from)
    else:
        checkpoint = {}

    # Build the SimCLR model.
    model = SimCLR(backbone)
    ckpt = checkpoint.get("model", None)
    if not ckpt is None:
        print("Restoring model...")
        model.load_state_dict(ckpt)
    model = model.to(DEVICE)

    # Prepare transform that creates multiple random views for every image.
    transform = SimCLRTransform(
        input_size=224,
        cj_prob = 0.8,
        cj_strength = 1.0,
        cj_bright = 0.8,
        cj_contrast = 0.8,
        cj_sat = 0,
        cj_hue = 0,
        min_scale = 0.25,
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
    dataset = TarFLCDataset(
        tar_path="./data/FLCDataset/dataset.tar", transform=transform, 
        use_cache=True, cache_system=cache_system, max_cache_size=8e9)

    # Build a PyTorch dataloader.
    dataloader = torch.utils.data.DataLoader(
        dataset,  # Pass the dataset to the dataloader.
        batch_size=128,  # A large batch size helps with the learning.
        shuffle=True,  # Shuffling is important!
        num_workers=4
    )

    # Lightly exposes building blocks such as loss functions.
    criterion = loss.NTXentLoss(temperature=0.5)

    # Get a PyTorch optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=2e-5)
    ckpt = checkpoint.get("optimizer", None)
    if not ckpt is None:
        print("Restoring optimizer...")
        optimizer.load_state_dict(ckpt)

    # Train the model.
    stats = defaultdict(list)
    ckpt = checkpoint.get("stats", None)
    if not ckpt is None:
        stats = ckpt
    for epoch in range(len(stats["mean"]), 512):
        print(f"[----] Epoch: {epoch}")
        pbar = tqdm(dataloader, leave=False)
        stats_loss, running_loss, running_batches = [], 0, 0
        for (view0, view1) in pbar:

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

        print(f"[----] Epoch: {epoch}")
        for key, func in zip(["mean", "std", "min", "max", "median"], 
                            [numpy.mean, numpy.std, numpy.min, numpy.max, numpy.median]):
            stats[key].append(func(stats_loss))
        
        torch.save({
            "optimizer" : optimizer.state_dict(),
            "model" : model.state_dict(),
            "stats" : stats
        }, "./data/ssl/baselines/resnet18/result.pt")