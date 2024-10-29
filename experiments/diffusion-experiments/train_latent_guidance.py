import numpy as np
import matplotlib.pyplot as plt 
from lightning.pytorch import Trainer
from lightning.pytorch.core import LightningModule 
from lightning.pytorch.callbacks import ModelCheckpoint 
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
import torch 
from diffusion_models.diffusion.ddpm_lightning import DDPM 
from diffusion_models.diffusion.denoising.unet import UNet 
from tqdm import trange, tqdm 
from torch import nn
import os
import argparse
import sys 
from datamodule import MultiprocessingDataModule
from class_dict import class_dict
sys.path.insert(0, "../")
from model_builder import get_pretrained_model_v2
from utils import SaveBestModel, AverageMeter, compute_Nary_accuracy, track_loss, update_cfg, get_number_of_classes

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset-path", type=str, default="./Datasets/FLCDataset/baselines/dataset.tar")
parser.add_argument("--model", default="mae-lightning-tiny")
parser.add_argument("--weights", type=str, default="MAE_TINY_STED")
parser.add_argument("--timesteps", type=int, default=1000)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--dataset", type=str, default="STED")
parser.add_argument("--save-folder", type=str, default='./model-checkpoints/latent')
parser.add_argument("--num-classes", type=int, default=24)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--use-tensorboard", action='store_true')
parser.add_argument("--restore-from", type=str, default=None)
args = parser.parse_args() 


class ReconstructionCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        imgs = pl_module.imgs # We've defined this in the training_step function of the LightningModule 
        pl_module.eval()
        with torch.no_grad():
            conditions = pl_module.latent_encoder.forward_features(imgs)
            samples = pl_module.p_sample_loop(shape=imgs.shape, con=conditions, progress=True)
            for i in range(samples.shape[0]):
                img = imgs[i].squeeze().detach().cpu().numpy()
                sample = samples[i].squeeze().detach().cpu().numpy()# .reshape(64, 64, 1)
                m, M = sample.min(), sample.max()
                sample = (sample - m) / (M - m)
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(img, cmap='hot', vmin=0.0, vmax=1.0)
                axs[1].imshow(sample, cmap='hot', vmin=0.0, vmax=1.0)
                axs[0].set_title("Original")
                axs[1].set_title("Reconstruction")
                axs[1].set_xticks([])
                axs[1].set_yticks([])
                fig.savefig(f"./model-checkpoints/latent/pl-latent-guidance-recon_{i}.png", dpi=1200)
                plt.close(fig)

class DatasetConfig:
    num_workers : int = None
    shuffle : bool = True
    use_cache : bool = True
    max_cache_size : float = 32e+9
    return_metadata : bool = True
    batch_size: int = args.batch_size

if __name__=="__main__":
    if args.checkpoint is not None:
        raise NotImplementedError("Loading from checkpoint not implemented yet")

    else:
        OUTPUT_FOLDER = args.save_folder 
        latent_encoder, model_config = get_pretrained_model_v2(
            name=args.model,
            weights=args.weights,
            path=None,
            mask_ratio=0.0, 
            pretrained=False,
            in_channels=1,
            as_classifier=True,
            blocks="all",
            num_classes=4, # will not be used
        )
        denoising_model = UNet(
            dim=64, 
            channels=1, 
            cond_dim=model_config.dim,
            dim_mults=(1,2,4),
            condition_type="latent",
            num_classes=4 # placeholder, not used
        )
        model = DDPM(
            denoising_model=denoising_model,
            timesteps=args.timesteps,
            beta_schedule="linear",
            condition_type="latent",
            latent_encoder=latent_encoder
        )

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    logger = TensorBoardLogger(OUTPUT_FOLDER) if args.use_tensorboard else None
    last_model_callback = ModelCheckpoint(
        dirpath=OUTPUT_FOLDER,
        every_n_epochs=1,
        filename="current_model",
        enable_version_counter=False
    )
    last_model_callback.FILE_EXTENSION = ".pth"

    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_FOLDER,
        every_n_epochs=10,
        filename="checkpoint-{epoch}",
        save_top_k=-1,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )
    checkpoint_callback.FILE_EXTENSION = ".pth"
    
    callbacks = [last_model_callback, checkpoint_callback, ReconstructionCallback()]
    cfg = DatasetConfig()
    datamodule = MultiprocessingDataModule(args, cfg, transform=None)
    trainer = Trainer(
        max_epochs=1000,
        devices='auto',
        accelerator='gpu',
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
        strategy="ddp_find_unused_parameters_true",
        sync_batchnorm=True,
        use_distributed_sampler=False,
        logger=logger,
        callbacks=callbacks
    )
    trainer.fit(model, train_dataloaders=datamodule, ckpt_path=args.restore_from)
