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


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--restore-from", type=str, default="")
parser.add_argument("--model", type=str, default='mae-small')
parser.add_argument("--save-folder", type=str, default='./Datasets/FLCDataset/baselines/mae_small_STED')
parser.add_argument("--dataset-path", type=str, default="./Datasets/FLCDataset/dataset.tar")
parser.add_argument("--modality", type=str, default="STED", choices=["STED", "JUMP"])
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

   
    logger = TensorBoardLogger(OUTPUT_FOLDER) if args.use_tensorboard else None
    
    MAETransform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.ToTensor(),
        RandomResizedCropMinimumForeground(size=224, scale=(0.3, 1.0)),
    ])

    last_model_callback = ModelCheckpoint(
        dirpath=OUTPUT_FOLDER,
        every_n_epochs=1,
        filename="pl_current_model",
        enable_version_counter=False
    )
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


    manager = Manager()
    cache_system = manager.dict()
    dataset = get_dataset(
        name=args.dataset,
        path=args.dataset_path,
        transform=MAETransform,
        use_cache=False,
        cache_system=cache_system,
        max_cache_size=16e9,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=8
    )

    trainer = Trainer(
        max_epochs=1600,
        devices='auto',
        accelerator='gpu',
        strategy='ddp',
        sync_batchnorm=True,
        use_distributed_sampler=True,
        logger=logger,
        callbacks=callbacks
    )

    trainer.fit(model, train_dataloaders=dataloader, ckpt_path=args.restore_from)



