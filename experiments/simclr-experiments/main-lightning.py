
import math
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
from lightly.models.utils import get_weight_decay_parameters
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler
from lightly.utils.benchmarking import MetricCallback
from torch.optim import SGD
from typing import Tuple, Any
from dataclasses import dataclass

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.core import LightningModule, LightningDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, DeviceStatsMonitor, EarlyStopping, LearningRateMonitor

from tqdm import tqdm
from collections import defaultdict
from collections.abc import Mapping
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import sys 
sys.path.insert(0, "..")
from configuration import Configuration
from modules.datamodule import MultiprocessingDataModule
from modules.transforms import SimCLRTransform
from modules.loss import NTXentLossWithClasses
from model_builder import get_base_model
from utils import update_cfg

# Define the configuration for the SimCLR model.
class SimCLRTransformConfig(Configuration):

    input_size : int = 224
    cj_prob : float = 0.8
    cj_strength : float = 1.0
    cj_bright : float = 0.8
    cj_contrast : float = 0
    cj_sat : float = 0
    cj_hue : float = 0
    cj_gamma : float = 0
    scale : Tuple[float, float] = (1.0, 1.0)
    random_gray_scale : float = 0
    gaussian_blur : float = 0
    kernel_size : float = None
    sigmas : Tuple[float, float] = (0.1, 2)
    vf_prob : float = 0.5
    hf_prob : float = 0.5
    rr_prob : float = 0.5
    rr_degrees : float = None
    normalize : bool = False
    gaussian_noise_prob : float = 0.5
    gaussian_noise_mu: float = 0.
    gaussian_noise_std: float = 0.05
    poisson_noise_prob : float = 0.5
    poisson_noise_lambda : float = 0.5  

class DataModuleConfig(Configuration):

    num_workers : int = None
    shuffle : bool = True
    use_cache : bool = True
    max_cache_size : float = 32e+9
    return_metadata : bool = True

class SimCLRConfig(Configuration):

    histogram_every_n_epochs : int = 10
    hidden_dim : int = 2048
    output_dim : int = 128
    temperature : float = 0.1
    lr : float = 0.3
    lr_scaling : str = None
    momentum : float = 0.9
    weight_decay : float = 1e-4
    eps : float = 1e-8
    warmup : float = 0.01

# Create a PyTorch module for the SimCLR model.
class SimCLR(LightningModule):
    def __init__(self, cfg, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.cfg = cfg
        self.backbone, _ = get_base_model(self.cfg.args.backbone)

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.projection_head = heads.SimCLRProjectionHead(
            input_dim=self.cfg.dim,
            hidden_dim=self.cfg.simclr.hidden_dim,
            output_dim=self.cfg.simclr.output_dim,
        )

        self.criterion = NTXentLossWithClasses(temperature=self.cfg.simclr.temperature, gather_distributed=True)

    def on_train_epoch_end(self):

        if self.current_epoch % self.cfg.simclr.histogram_every_n_epochs == 0:
            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # Makes sure the configuration is converted to a dictionary
        checkpoint["hyper_parameters"]["cfg"] = checkpoint["hyper_parameters"]["cfg"].to_dict()                

    def training_step(self, batch, batch_idx):
        view0, view1 = batch
        metadata = None
        if isinstance(view1, dict):
            metadata = view1
            view0, view1 = view0

            # We simply convert to int using the hash method.
            metadata = torch.tensor([hash(m) for m in metadata["path"]], dtype=torch.long, device=view0.device)

        z0 = self.forward(view0)
        z1 = self.forward(view1)

        loss = self.criterion(z0, z1, metadata=metadata)

        # Logging
        self.log("train_loss", loss, sync_dist=True, prog_bar=True, batch_size=len(view0))
        self.log("Loss/mean", loss, sync_dist=True, prog_bar=False)
        self.log("Loss/min", loss, reduce_fx=torch.min, sync_dist=True)
        self.log("Loss/max", loss, reduce_fx=torch.max, sync_dist=True)

        # logging average activations
        self.log("activations/z0", z0.mean(), sync_dist=True, prog_bar=False, batch_size=len(view0))
        self.log("activations/z1", z1.mean(), sync_dist=True, prog_bar=False, batch_size=len(view0))
        self.log("activations/z0-std", z0.std(dim=1).mean(), sync_dist=True, prog_bar=False, batch_size=len(view0))
        self.log("activations/z1-std", z1.std(dim=1).mean(), sync_dist=True, prog_bar=False, batch_size=len(view0))

        # Logging images
        writer = self.logger.experiment
        if (batch_idx == 0) and (self.current_epoch % 10 == 0) and isinstance(writer, SummaryWriter):
            writer.add_images("Images/view0", view0[:16], self.current_epoch, dataformats="NCHW")
            writer.add_images("Images/view1", view1[:16], self.current_epoch, dataformats="NCHW")        

        return loss

    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head]
        )

        lr = self.cfg.simclr.lr 
        if self.cfg.simclr.lr_scaling == "linear":
            lr = lr * self.cfg.batch_size * self.trainer.world_size / 256
        elif self.cfg.simclr.lr_scaling == "sqrt":
            lr = lr * math.sqrt(self.cfg.batch_size * self.trainer.world_size)
         
        optimizer = LARS(
            [
                {
                    "name": "simclr", 
                    "params": params
                },
                {
                    "name": "simclr_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                }
            ],
            # Square root learning rate scaling improves performance for small
            # batch sizes (<=2048) and few training epochs (<=200). Alternatively,
            # linear scaling can be used for larger batches and longer training:
            # lr = 0.3 * self.cfg.batch_size * self.trainer.world_size / 256,
            # See Appendix B.1. in the SimCLR paper https://arxiv.org/abs/2002.05709
            # lr=0.075 * math.sqrt(self.cfg.batch_size * self.trainer.world_size),
            # lr=0.075 * math.sqrt(1024),
            lr = lr,
            # lr = 0.3 / self.trainer.world_size,
            # lr = 0.3,
            momentum = self.cfg.simclr.momentum,
            # Note: Paper uses weight decay of 1e-6 but reference code 1e-4. See:
            # https://github.com/google-research/simclr/blob/2fc637bdd6a723130db91b377ac15151e01e4fc2/README.md?plain=1#L103
            weight_decay = self.cfg.simclr.weight_decay,
            eps = self.cfg.simclr.eps, # In 16-mixed training, eps 1e-8 is 0.
        )
        print("-----Optimizer-----")
        print(f"{self.trainer.estimated_stepping_batches=}")
        print(f"{self.trainer.max_epochs=}")
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=int(
                    self.trainer.estimated_stepping_batches * self.cfg.simclr.warmup
                    # / self.trainer.max_epochs
                    # * 10
                ),
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

    def forward(self, x):
        features = self.backbone(x)
        if features.dim() > 2:
            features = self.avg_pool(features)
        features = torch.flatten(features, start_dim=1)
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

    seed_everything(args.seed, workers=True)
    torch.set_flush_denormal(True)

    # Assert args.opts is a multiple of 2
    if len(args.opts) == 1:
        args.opts = args.opts[0].split(" ")
    assert len(args.opts) % 2 == 0, "opts must be a multiple of 2"    

    backbone, cfg = get_base_model(args.backbone)
    cfg.transform = SimCLRTransformConfig()
    cfg.datamodule = DataModuleConfig()
    cfg.simclr = SimCLRConfig()
    cfg.args = args
    update_cfg(cfg, args.opts)

    # Return metadata should be activated only in case of STED dataset
    cfg.datamodule.return_metadata = cfg.datamodule.return_metadata and args.dataset == "STED"
   
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
        every_n_epochs=10,
        enable_version_counter=False
    )
    last_model_callback.FILE_EXTENSION = ".pt"
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=OUTPUT_FOLDER,
    #     every_n_epochs=10,
    #     filename="checkpoint-{epoch}",
    #     save_top_k=-1,
    #     auto_insert_metric_name=False,
    #     enable_version_counter=False
    # )
    # checkpoint_callback.FILE_EXTENSION = ".pt"
    step_checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_FOLDER,
        every_n_train_steps=5000,
        filename="checkpoint-{step}",
        save_top_k=-1,
        auto_insert_metric_name=False,
        enable_version_counter=False
    )
    step_checkpoint_callback.FILE_EXTENSION = ".pt"

    # metric_callback = MetricCallback()
    callbacks = [
        LearningRateMonitor(),
        EarlyStopping(monitor="train_loss", patience=int(1e12), check_finite=True),
        # DeviceStatsMonitor(),
        # metric_callback,
        last_model_callback, 
        # checkpoint_callback,
        step_checkpoint_callback
    ]

    # Build the SimCLR model.
    model = SimCLR(cfg)
    if args.restore_from:
        print("Restoring model...")
        model = SimCLR.load_from_checkpoint(args.restore_from, cfg=cfg)

    summary(model, input_size=(1, 224, 224), device=model.device.type)

    # Prepare transform that creates multiple random views for every image.
    transform = SimCLRTransform(
        input_size = cfg.transform.input_size,
        cj_prob = cfg.transform.cj_prob,
        cj_strength = cfg.transform.cj_strength,
        cj_bright = cfg.transform.cj_bright,
        cj_contrast = cfg.transform.cj_contrast,
        cj_sat = cfg.transform.cj_sat,
        cj_hue = cfg.transform.cj_hue,
        cj_gamma = cfg.transform.cj_gamma,
        scale = cfg.transform.scale,
        random_gray_scale = cfg.transform.random_gray_scale,
        gaussian_blur = cfg.transform.gaussian_blur,
        kernel_size = cfg.transform.kernel_size,
        sigmas = cfg.transform.sigmas,
        vf_prob = cfg.transform.vf_prob,
        hf_prob = cfg.transform.hf_prob,
        rr_prob = cfg.transform.rr_prob,
        rr_degrees = cfg.transform.rr_degrees,
        normalize = cfg.transform.normalize,
        gaussian_noise_prob = cfg.transform.gaussian_noise_prob,
        gaussian_noise_mu = cfg.transform.gaussian_noise_mu,
        gaussian_noise_std = cfg.transform.gaussian_noise_std,
        poisson_noise_prob = cfg.transform.poisson_noise_prob,
        poisson_noise_lambda = cfg.transform.poisson_noise_lambda
    )

    datamodule = MultiprocessingDataModule(args, cfg, transform=transform, debug=args.dry_run)

    trainer = Trainer(
        max_epochs=-1,
        max_steps=1_000_000,
        devices="auto",
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
        accelerator="gpu",
        # precision="16-mixed",
        gradient_clip_val=1.0,
        strategy="ddp",
        sync_batchnorm=True,
        use_distributed_sampler=False,
        logger=logger,
        callbacks=callbacks,
        # detect_anomaly=True,
    )
    trainer.fit(model, train_dataloaders=datamodule, ckpt_path=args.restore_from)
