
import math
import copy
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
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import get_weight_decay_parameters, deactivate_requires_grad, update_momentum
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule
from lightly.utils.benchmarking import MetricCallback
from torch.optim import SGD, Adam
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
from DEFAULTS import BASE_PATH
from configuration import Configuration
from modules.datamodule import MultiprocessingDataModule
from modules.transforms import DINOTransform
from modules.loss import NTXentLossWithClasses
from model_builder import get_base_model
from utils import update_cfg

# Define the configuration for the DINO model.
class DINOTransformConfig(Configuration):

    global_crop_size : int = 224
    global_crop_scale : Tuple[float, float] = (1.0, 1.0)
    local_crop_size : int = 224
    local_crop_scale : Tuple[float, float] = (1.0, 1.0)
    n_local_views: int = 2
    cj_prob : float = 0.8
    cj_strength : float = 1.0
    cj_bright : float = 0.8
    cj_contrast : float = 0
    cj_sat : float = 0
    cj_hue : float = 0
    cj_gamma : float = 0
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
    return_metadata : bool = False

class DINOConfig(Configuration):

    histogram_every_n_epochs : int = 10
    hidden_dim : int = 512
    bottleneck_dim : int = 64
    output_dim : int = 2048
    freeze_last_layer : bool = True
    temperature : float = 0.1
    lr : float = 0.0005
    lr_scaling : str = "linear"
    weight_decay : float = 1e-4
    eps : float = 1e-8
    warmup : float = 0.01

# Create a PyTorch module for the DINO model.
class DINO(LightningModule):
    def __init__(self, cfg, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.cfg = cfg
        self.student_backbone, _ = get_base_model(self.cfg.args.backbone)
        self.student_head = DINOProjectionHead(
            input_dim=self.cfg.dim,
            hidden_dim=self.cfg.dino.hidden_dim,
            output_dim=self.cfg.dino.output_dim,
            bottleneck_dim=self.cfg.dino.bottleneck_dim,
            freeze_last_layer=self.cfg.dino.freeze_last_layer,
        )

        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_head = DINOProjectionHead(
            input_dim=self.cfg.dim,
            hidden_dim=self.cfg.dino.hidden_dim,
            output_dim=self.cfg.dino.output_dim,
            bottleneck_dim=self.cfg.dino.bottleneck_dim,
        )
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)

        self.criterion = loss.DINOLoss(
            output_dim=self.cfg.dino.output_dim,
        )

    def on_train_epoch_end(self):

        if self.current_epoch % self.cfg.dino.histogram_every_n_epochs == 0:
            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # Makes sure the configuration is converted to a dictionary
        checkpoint["hyper_parameters"]["cfg"] = checkpoint["hyper_parameters"]["cfg"].to_dict()

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(
            self.current_epoch, 
            int(self.trainer.estimated_stepping_batches // self.cfg.batch_size), 
            0.996, 1
        )
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)

        views = batch
        global_views = views[:2]

        # Logging images
        writer = self.logger.experiment
        if (batch_idx == 0) and (self.current_epoch % 10 == 0) and isinstance(writer, SummaryWriter):
            writer.add_images("Images/global_views_0", global_views[0][:16], self.current_epoch, dataformats="NCHW")
            writer.add_images("Images/global_views_1", global_views[1][:16], self.current_epoch, dataformats="NCHW")
            # writer.add_images("Images/local_views_0", views[2][:16], self.current_epoch, dataformats="NCHW")
            
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)

        # Logging
        self.log("train_loss", loss, sync_dist=True, prog_bar=True, batch_size=len(global_views[0]))
        self.log("Loss/mean", loss, sync_dist=True, prog_bar=False)
        self.log("Loss/min", loss, reduce_fx=torch.min, sync_dist=True)
        self.log("Loss/max", loss, reduce_fx=torch.max, sync_dist=True)

        return loss

    def configure_optimizers(self):

        lr = self.cfg.dino.lr 
        if self.cfg.dino.lr_scaling == "linear":
            lr = lr * self.cfg.batch_size * self.trainer.world_size / 256

        optimizer = Adam(
            self.parameters(),
            lr = lr,
            weight_decay = self.cfg.dino.weight_decay,
            eps = self.cfg.dino.eps, # In 16-mixed training, eps 1e-8 is 0.
        )
        print("-----Optimizer-----")
        print(f"{self.trainer.estimated_stepping_batches=}")
        print(f"{self.trainer.max_epochs=}")
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=int(
                    self.trainer.estimated_stepping_batches * self.cfg.dino.warmup
                ),
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

    def forward(self, x):
        features = self.student_backbone(x)
        if features.dim() > 2:
            features = self.avg_pool(features)
        features = torch.flatten(features, start_dim=1)
        z = self.student_head(features)
        return z

    def forward_teacher(self, x):
        features = self.teacher_backbone(x)
        if features.dim() > 2:
            features = self.avg_pool(features)
        features = torch.flatten(features, start_dim=1)
        z = self.teacher_head(features)
        return z

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")     
    parser.add_argument("--restore-from", type=str, default=None,
                    help="Model from which to restore from") 
    parser.add_argument("--save-folder", type=str, default=f"{BASE_PATH}/baselines",
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
    cfg.transform = DINOTransformConfig()
    cfg.datamodule = DataModuleConfig()
    cfg.dino = DINOConfig()
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
        every_n_epochs=10,
        enable_version_counter=False
    )
    last_model_callback.FILE_EXTENSION = ".pt"
    step_checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_FOLDER,
        every_n_train_steps=5000,
        filename="checkpoint-{step}",
        save_top_k=-1,
        auto_insert_metric_name=False,
        enable_version_counter=False
    )
    step_checkpoint_callback.FILE_EXTENSION = ".pt"

    callbacks = [
        LearningRateMonitor(),
        EarlyStopping(monitor="train_loss", patience=int(1e12), check_finite=True),
        last_model_callback, 
        step_checkpoint_callback
    ]

    # Build the DINO model.
    model = DINO(cfg)
    if args.restore_from:
        print("Restoring model...")
        model = DINO.load_from_checkpoint(args.restore_from, cfg=cfg)

    summary(model, input_size=(1, 224, 224), device=model.device.type)

    # Prepare transform that creates multiple random views for every image.
    transform = DINOTransform(
        global_crop_size = cfg.transform.global_crop_size,
        global_crop_scale=cfg.transform.global_crop_scale,
        local_crop_size = cfg.transform.local_crop_size,
        local_crop_scale=cfg.transform.local_crop_scale,
        n_local_views=cfg.transform.n_local_views,
        cj_prob = cfg.transform.cj_prob,
        cj_strength = cfg.transform.cj_strength,
        cj_bright = cfg.transform.cj_bright,
        cj_contrast = cfg.transform.cj_contrast,
        cj_sat = cfg.transform.cj_sat,
        cj_hue = cfg.transform.cj_hue,
        cj_gamma = cfg.transform.cj_gamma,
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
        gradient_clip_val=1.0,
        strategy="ddp",
        sync_batchnorm=True,
        use_distributed_sampler=False,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model, train_dataloaders=datamodule, ckpt_path=args.restore_from)
