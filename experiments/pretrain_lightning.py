import torch 
import os 
import argparse 
import torchvision.transforms
from loaders import get_STED_dataset, get_JUMP_dataset
from collections import defaultdict 
from collections.abc import Mapping 
from multiprocessing import Manager 
from torch.utils.tensorboard import SummaryWriter
from lightning.pytorch import Trainer
from lightning.pytorch.core import LightningModule 
from lightning.pytorch.callbacks import ModelCheckpoint 
from lightning.pytorch.loggers import TensorBoardLogger
from tqdm import tqdm 
from model_builder import get_base_model 
from torchinfo import summary 
from models.lightly_mae import MAE
from datasets import get_dataset
from modules.transforms import RandomResizedCropMinimumForeground
from modules.datamodule import MultiprocessingDataModule


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, default="STED")
parser.add_argument("--restore-from", type=str, default=None)
parser.add_argument("--model", type=str, default='mae-small')
parser.add_argument("--save-folder", type=str, default='./Datasets/FLCDataset/baselines/mae_small_STED')
parser.add_argument("--dataset-path", type=str, default="./Datasets/FLCDataset/dataset.tar")
parser.add_argument("--use-tensorboard", action='store_true')
args = parser.parse_args()


if __name__=="__main__":
    model, cfg = get_base_model(name=args.model)

    if args.restore_from:
        OUTPUT_FOLDER = os.path.dirname(args.restore_from)
        print("--- Restoring model ---")
        model = MAE.load_from_checkpoint(args.restore_from)
    else:
        OUTPUT_FOLDER = args.save_folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"--- Loaded model {args.model} successfully ---")

   
    logger = TensorBoardLogger(OUTPUT_FOLDER) if args.use_tensorboard else None
    
    MAETransform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        # RandomResizedCropMinimumForeground(size=224, scale=(0.3, 1.0)),
    ])

    last_model_callback = ModelCheckpoint(
        dirpath=OUTPUT_FOLDER,
        every_n_epochs=1,
        filename="pl_current_model",
        enable_version_counter=False
    )
    last_model_callback.FILE_EXTENSION = ".pth"
    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_FOLDER,
        every_n_epochs=10,
        filename="pl_checkpoint-{epoch}",
        save_top_k=-1,
        auto_insert_metric_name=False,
        enable_version_counter=False
    )
    checkpoint_callback.FILE_EXTENSION = ".pth"
    callbacks = [last_model_callback, checkpoint_callback]

    datamodule = MultiprocessingDataModule(args, cfg, transform=MAETransform)

    trainer = Trainer(
        max_epochs=1600,
        devices='auto',
        accelerator='gpu',
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
        strategy='ddp_find_unused_parameters_true',
        sync_batchnorm=True,
        use_distributed_sampler=True,
        logger=logger,
        callbacks=callbacks
    )

    trainer.fit(model, train_dataloaders=datamodule, ckpt_path=args.restore_from)



