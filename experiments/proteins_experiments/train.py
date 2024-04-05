import torch
import torchvision 
import numpy as np
import os
from typing import List
import random
from lightly import loss, transforms
from lightly.data import LightlyDataset
from lightly.models.modules import heads
from tqdm import tqdm 
from collections import defaultdict
from multiprocessing import Manager 
from collections.abc import Mapping
from torch.utils.tensorboard import SummaryWriter 
from torchinfo import summary 
from models.backbones import resnet
from datasets.datasets import TarFLCDataset



class SimCLR(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone == backbone
        if isinstance(self.backbone, torchvision.models.resnet.ResNet):
            dim = 512
        else:
            dim = 512
        self.projection_head = heads.SimCLRProjectionHead(
            input_dim=512, 
            hidden_dim=512,
            output_dim=128
        )

    def load_state_dict(self, state_dict: Mapping[str, torch.Any], strict: bool = True, assign: bool = False):
        self.backbone.load_state_dict(state_dict["backbone"])
        self.projection_head.load_state_dict(state_dict["projection-head"])

    def state_dict(self):
        return {
            "backbone": self.backbone.state_dict(),
            "projection-head": self.projection_head.state_dict()
        }
    
    def forward(self, x: torch.Tensor):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)
        return z
    
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
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
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    backbone, cfg = resnet.get_backbone(args.backbone)

    if args.restore_from:
        checkpoint = torch.load(args.restore_from)
        OUTPUT_FOLDER = os.path.dirname(args.restore_from)
    else:
        checkpoint = {}
        OUTPUT_FOLDER = os.path.join(args.save_folder, args.backbone)
    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(OUTPUT_FOLDER, "logs"))

    model = SimCLR(backbone)
    ckpt = checkpoint.get("model", None)

    if not ckpt is None:
        print("Restoring model...")
        model.load_state_dict(ckpt)
    model = model.to(DEVICE)

    summary(model, input_size=(256, 1, 64, 64))

    # transform = SimCLRTransform TODO

    manager = Manager()
    cache_system = manager.dict()
    tar_path = args.dataset_path
    # dataset = TarFLCDataset(
    #     tar_path=tar_path,
    #     transform=transform,
    #     use_cache=True,
    #     cache_system=cache_system,
    #     max_cache_size=16e9
    # ) TODO
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=6,
        drop_last=False
    )