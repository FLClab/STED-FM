
import torch

from multiprocessing import Manager
from lightning.pytorch.core import LightningDataModule

import sys 
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