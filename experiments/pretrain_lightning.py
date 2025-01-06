import torch 
import os 
import argparse 
import torchvision.transforms
from loaders import get_STED_dataset, get_JUMP_dataset
from collections import defaultdict 
from collections.abc import Mapping 
from multiprocessing import Manager 
from torch.utils.tensorboard import SummaryWriter
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.core import LightningModule 
from lightning.pytorch.callbacks import ModelCheckpoint 
from lightning.pytorch.loggers import TensorBoardLogger
from tqdm import tqdm 
from model_builder import get_base_model 
# from torchinfo import summary 

from DEFAULTS import BASE_PATH
from configuration import Configuration
from models.lightly_mae import MAE
from datasets import get_dataset
from modules.transforms import RandomResizedCropMinimumForeground
from modules.datamodule import MultiprocessingDataModule

import sys
sys.path.insert(0, ".")
from utils import update_cfg


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, default="STED")
parser.add_argument("--restore-from", type=str, default=None)
parser.add_argument("--model", type=str, default='mae-small')
parser.add_argument("--save-folder", type=str, default=f"{BASE_PATH}/baselines")
parser.add_argument("--dataset-path", type=str, default="./Datasets/FLCDataset/dataset.tar")
### Used only with Hybrid dataset
parser.add_argument("--hpa-path", type=str, default=None)
parser.add_argument("--sim-path", type=str, default=None)
parser.add_argument("--sted-path", type=str, default=None)
### 
parser.add_argument("--use-tensorboard", action='store_true')
parser.add_argument("--opts", nargs="+", default=[], 
                    help="Additional configuration options")    
parser.add_argument("--dry-run", action="store_true",
                    help="Activates dryrun")        
args = parser.parse_args()
    
# Assert args.opts is a multiple of 2
if len(args.opts) == 1:
    args.opts = args.opts[0].split(" ")
assert len(args.opts) % 2 == 0, "opts must be a multiple of 2"    

class DataModuleConfig(Configuration):

    num_workers : int = None
    shuffle : bool = True
    use_cache : bool = True
    max_cache_size : float = 32e+9
    return_metadata : bool = False

if __name__=="__main__":

    seed_everything(args.seed, workers=True)

    model, cfg = get_base_model(name=args.model)
    cfg.datamodule = DataModuleConfig()
    cfg.args = args
    update_cfg(cfg, args.opts)

    if args.restore_from:
        OUTPUT_FOLDER = os.path.dirname(args.restore_from)
        print("--- Restoring model ---")
        in_channels = cfg.in_channels
        model = MAE.load_from_checkpoint(args.restore_from, vit=model.backbone.vit, in_channels=cfg.in_channels, mask_ratio=cfg.mask_ratio)
        print(f"--- Restored model {args.model}  from {args.restore_from} successfully ---")
    else:
        OUTPUT_FOLDER = args.save_folder
        # print(f"--- Exiting while debugging the restore-from argument ---") # TODO: Remove this once the restore-from argument is working
        # exit()
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"--- Loaded model {args.model} successfully ---")

    logger = TensorBoardLogger(OUTPUT_FOLDER) if args.use_tensorboard else None
    
    MAETransform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        RandomResizedCropMinimumForeground(size=224, scale=(1.0, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
    ])

    last_model_callback = ModelCheckpoint(
        dirpath=OUTPUT_FOLDER,
        every_n_epochs=1,
        filename="current_model",
        enable_version_counter=False
    )
    last_model_callback.FILE_EXTENSION = ".pth"
    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_FOLDER,
        every_n_epochs=100,
        filename="checkpoint-{epoch}",
        save_top_k=-1,
        auto_insert_metric_name=False,
        enable_version_counter=False
    )
    checkpoint_callback.FILE_EXTENSION = ".pth"
    callbacks = [last_model_callback, checkpoint_callback]

    datamodule = MultiprocessingDataModule(args, cfg, transform=MAETransform, debug=args.dry_run)

    trainer = Trainer(
        max_epochs=1000,
        devices='auto',
        accelerator='gpu',
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
        strategy='ddp_find_unused_parameters_true',
        sync_batchnorm=True,
        use_distributed_sampler=False,
        logger=logger,
        callbacks=callbacks
    )

    trainer.fit(model, train_dataloaders=datamodule, ckpt_path=args.restore_from)