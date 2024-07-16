
import torch

from multiprocessing import Manager
from lightning.pytorch.core import LightningDataModule

import sys

import torch.distributed 
sys.path.insert(0, "..")
from datasets import get_dataset

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

        self.dataset_name = args.dataset
        self.dataset_path = args.dataset_path
        self.kwargs = kwargs

    def setup(self, stage : str = None):
        try:
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        except AssertionError as err:
            self.world_size = 1
            self.rank = 0

        # Builds one dataset per process
        manager = Manager()
        cache_system = manager.dict()
        self.dataset = get_dataset(
            self.dataset_name, self.dataset_path, 
            use_cache=True, cache_system=cache_system, 
            max_cache_size=8e9,
            world_size = self.world_size, rank = self.rank,
            **self.kwargs
        )        
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset, 
            batch_size = self.cfg.batch_size,
            shuffle=True,
            num_workers=10,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=False,
            drop_last=True,
        )