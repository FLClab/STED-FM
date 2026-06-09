
import torch 
import os 
import argparse 
import random
import torchvision.transforms
from collections import defaultdict 
from collections.abc import Mapping 
from multiprocessing import Manager 
import matplotlib.pyplot as plt 
from torch.utils.tensorboard import SummaryWriter
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.core import LightningModule 
from lightning.pytorch.callbacks import ModelCheckpoint 
from lightning.pytorch import Callback
from lightning.pytorch.loggers import TensorBoardLogger
from tqdm import tqdm 
from stedfm.model_builder import get_base_model, get_pretrained_model_v2
# from torchinfo import summary 

from stedfm.DEFAULTS import BASE_PATH
from stedfm.configuration import Configuration
from stedfm.models.lightly_mae import MAE, MCMSMAE
from stedfm.datasets import get_dataset
from stedfm.modules.transforms import RandomResizedCropMinimumForeground, Random90DegreeRotation, SwapChannels
from stedfm.modules.datamodule import MultiprocessingDataModule
from stedfm.utils import update_cfg
from stedfm.datasets import ProteinImageDataset


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, default="MCSTED")
parser.add_argument("--restore-from", type=str, required=True)
parser.add_argument("--model", type=str, default='mae-mcms-lightning-small')
parser.add_argument("--dataset-path", type=str, default="./Datasets/FLCDataset/dataset.tar")
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
    use_cache : bool = False
    max_cache_size : float = 32e+9
    return_metadata : bool = True

class VariableRandomResizedCropMinimumForeground(torch.nn.Module):
    def __init__(self, min_size, max_size, multiple_of: int = 16):
        super().__init__()
        
        self.min_size = min_size
        self.max_size = max_size
        self.multiple_of = multiple_of

        self.possible_sizes = []
        for size in range(min_size, max_size + 1):
            if size % multiple_of == 0:
                self.possible_sizes.append(size)

    def forward(self, img):
        size = int(torch.randint(low=0, high=len(self.possible_sizes), size=(1,)).item())
        size = self.possible_sizes[size]
        transform = RandomResizedCropMinimumForeground(size=size, scale=(1.0, 1.0))
        return transform(img)

class UpdateTransformCallback(Callback):
    """
    PyTorch Lightning callback to update the datamodule's dataset transform at the start of each training epoch
    to include random resized cropping for data augmentation during training.

    Note. This callback does not update on the first epoch (epoch 0) because the datamodule is already initialized
          with a transform that includes random resized cropping.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.min_size = 128
        self.max_size = 512
        self.multiple_of = 16
        self.sizes = []
        for size in range(self.min_size, self.max_size + 1):
            if size % self.multiple_of == 0:
                self.sizes.append(size)

        self.initial_batch_size = None
        self.initial_size = 224

    def on_train_epoch_start(self, trainer, pl_module):
        if self.initial_batch_size is None:
            self.initial_batch_size = trainer.datamodule.cfg.batch_size

        size = random.choice(self.sizes)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            RandomResizedCropMinimumForeground(size=size, scale=(1.0, 1.0)),
            SwapChannels(p=0.5),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            Random90DegreeRotation(),
        ])
        datamodule = trainer.datamodule
        datamodule.dataset.set_transforms(transform)

        # Update batch size based on the new size to keep the same number of pixels per batch
        if self.initial_size / size < 1:
            new_batch_size = max(1, self.initial_batch_size * (self.initial_size / size)**2)
        else:
            new_batch_size = max(1, self.initial_batch_size * (self.initial_size / size))
        new_batch_size = int(new_batch_size)
        def nearest_power_of_2(n):
            """Returns the nearest power of 2 less than or equal to n."""
            if n < 1:
                return 1
            power = 1
            while power * 2 <= n:
                power *= 2
            return power
        new_batch_size = nearest_power_of_2(new_batch_size)
        datamodule.cfg.batch_size = new_batch_size


if __name__=="__main__":

    seed_everything(args.seed, workers=True)
    print(args)
    model, cfg = get_base_model(name=args.model, opts=args.opts)

    cfg.datamodule = DataModuleConfig()
    cfg.args = args
    update_cfg(cfg, args.opts)

    if args.restore_from:
        OUTPUT_FOLDER = os.path.dirname(args.restore_from)

        # We update the OUTPUT_FOLDER to mention the finetuning
        OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "image-scale-finetuning")

        print("--- Restoring model ---")
        in_channels = cfg.in_channels

        # We only load the state_dict of the model since we want the training to be independent from the pretraining
        checkpoint = torch.load(args.restore_from)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        print(f"--- Restored model {args.model}  from {args.restore_from} successfully ---")
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"--- Loaded model {args.model} successfully ---")

    logger = TensorBoardLogger(OUTPUT_FOLDER) if args.use_tensorboard else None
    
    MAETransform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        RandomResizedCropMinimumForeground(size=224, scale=(1.0, 1.0)),
        SwapChannels(p=0.5),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        Random90DegreeRotation(),
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
        every_n_epochs=10,
        filename="checkpoint-{epoch}",
        save_top_k=-1,
        auto_insert_metric_name=False,
        enable_version_counter=False
    )
    checkpoint_callback.FILE_EXTENSION = ".pth"
    callbacks = [last_model_callback, checkpoint_callback]
    callbacks.append(UpdateTransformCallback())

    datamodule = MultiprocessingDataModule(args, cfg, transform=MAETransform, debug=args.dry_run)

    trainer = Trainer(
        max_epochs=100,
        devices='auto',
        accelerator='gpu',
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
        strategy='ddp_find_unused_parameters_true',
        sync_batchnorm=True,
        use_distributed_sampler=False,
        logger=logger,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1 # This seems to be causing issues since the dataloader is in multiprocessing...
    )

    trainer.fit(model, train_dataloaders=datamodule)