
import torch
import torchvision
import numpy
import os
import typing
import random
import lightning

from lightly import loss
from lightly import transforms
from lightly.data import LightlyDataset
from lightly.models.modules import heads
from lightning.pytorch import Trainer
from lightning.pytorch.core import LightningModule, LightningDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary

from tqdm import tqdm
from collections import defaultdict
from collections.abc import Mapping
from multiprocessing import Manager
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from modules.transforms import SimCLRTransform

import sys 
sys.path.insert(0, "..")
from datasets import get_dataset
from model_builder import get_base_model
from utils import update_cfg

class MultiprocessingDataModule(LightningDataModule):
    """
    Implements a PyTorch Lightning DataModule that uses multiprocessing to load the data.

    This follows the implementation steps from
    https://lightning.ai/docs/pytorch/latest/advanced/training_tricks.html#sharing-datasets-across-process-boundaries
    """
    def __init__(self, args, cfg, **kwargs):
        """
        Instantiates the DataModule.

        :param args: The arguments passed to the script.
        :param cfg: The configuration object.
        """
        super(MultiprocessingDataModule, self).__init__()
        self.cfg = cfg
        manager = Manager()
        cache_system = manager.dict()
        self.dataset = get_dataset(args.dataset, args.dataset_path, use_cache=True, cache_system=cache_system, **kwargs)        
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset, 
            batch_size = self.cfg.batch_size,
            shuffle=True,
            num_workers=10,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

# Create a PyTorch module for the SimCLR model.
class SimCLR(LightningModule):
    def __init__(self, backbone, cfg, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.projection_head = heads.SimCLRProjectionHead(
            input_dim=self.cfg.dim,
            hidden_dim=512,
            output_dim=128,
        )

        self.criterion = loss.NTXentLoss(temperature=0.1, gather_distributed=True)

    def training_step(self, batch, batch_idx):
        view0, view1 = batch
        z0 = self.forward(view0)
        z1 = self.forward(view1)
        loss = self.criterion(z0, z1)

        # Logging
        self.log("Loss/mean", loss, on_epoch=True, sync_dist=True)
        self.log("Loss/min", loss, on_epoch=True, reduce_fx=torch.min, sync_dist=True)
        self.log("Loss/max", loss, on_epoch=True, reduce_fx=torch.max, sync_dist=True)

        # Logging images
        writer = self.logger.experiment
        if (batch_idx == 0) and isinstance(writer, SummaryWriter):
            writer.add_images("Images/view0", view0[:5], self.current_epoch, dataformats="NCHW")
            writer.add_images("Images/view1", view1[:5], self.current_epoch, dataformats="NCHW")        

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, weight_decay=1e-6)
        return optimizer

    def load_state_dict(self, state_dict: Mapping[str, torch.Any], strict: bool = True, assign: bool = False):
        self.backbone.load_state_dict(state_dict["backbone"])
        self.projection_head.load_state_dict(state_dict["projection-head"])
    
    def state_dict(self):
        return {
            "backbone" : self.backbone.state_dict(),
            "projection-head" : self.projection_head.state_dict()
        }

    def forward(self, x):
        features = self.backbone(x)
        if features.dim() > 2:
            features = self.avg_pool(features)
        z = self.projection_head(features)
        return z

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")     
    parser.add_argument("--restore-from", type=str, default=None,
                    help="Model from which to restore from") 
    parser.add_argument("--save-folder", type=str, default="./data/SSL/baselines",
                    help="Model from which to restore from")     
    parser.add_argument("--dataset", type=str, default="STED",
                    help="Model from which to restore from")         
    parser.add_argument("--dataset-path", type=str, default="./data/FLCDataset/20240214-dataset.tar",
                    help="Model from which to restore from")         
    parser.add_argument("--backbone", type=str, default="resnet18",
                        help="Backbone model to load")
    parser.add_argument("--use-tensorboard", action="store_true",
                        help="Logging using tensorboard")    
    parser.add_argument("--opts", nargs="+", default=[], 
                        help="Additional configuration options")    
    parser.add_argument("--dry-run", action="store_true",
                        help="Activates dryrun")        
    args = parser.parse_args()

    numpy.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Assert args.opts is a multiple of 2
    if len(args.opts) == 1:
        args.opts = args.opts[0].split(" ")
    assert len(args.opts) % 2 == 0, "opts must be a multiple of 2"    

    backbone, cfg = get_base_model(args.backbone)
    cfg.args = args
    update_cfg(cfg, args.opts)
   
    if args.restore_from:
        OUTPUT_FOLDER = os.path.dirname(args.restore_from)
    else:
        OUTPUT_FOLDER = os.path.join(args.save_folder, f"{args.backbone}_{args.dataset}")
    if args.dry_run:
        OUTPUT_FOLDER = os.path.join(args.save_folder, "debug")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    logger = None
    if args.use_tensorboard:
        logger = TensorBoardLogger(OUTPUT_FOLDER)

    # Callbacks
    last_model_callback = ModelCheckpoint(
        dirpath=OUTPUT_FOLDER,
        filename="result",
        every_n_epochs=1,
        enable_version_counter=False
    )
    last_model_callback.FILE_EXTENSION = ".pt"
    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_FOLDER,
        every_n_epochs=10,
        filename="checkpoint-{epoch}",
        save_top_k=-1,
        auto_insert_metric_name=False,
        enable_version_counter=False
    )
    checkpoint_callback.FILE_EXTENSION = ".pt"
    callbacks = [last_model_callback, checkpoint_callback]

    # Build the SimCLR model.
    model = SimCLR(backbone, cfg)
    if args.restore_from:
        print("Restoring model...")
        model = SimCLR.load_from_checkpoint(args.restore_from, backbone=backbone, cfg=cfg)

    summary(model, input_size=(1, 224, 224), device=model.device.type)

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

    datamodule = MultiprocessingDataModule(args, cfg, transform=transform)

    trainer = Trainer(
        max_epochs=1000,
        devices="auto",
        accelerator="gpu",
        strategy="ddp_spawn",
        sync_batchnorm=True,
        use_distributed_sampler=True,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model, train_dataloaders=datamodule, ckpt_path=args.restore_from)
