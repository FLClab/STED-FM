
import torch
import random
import os

from torch.utils.data import default_collate
from multiprocessing import Manager
from lightning.pytorch.core import LightningDataModule

import sys

import torch.distributed
from stedfm.datasets import get_dataset

class MultiprocessingDistributedSampler(torch.utils.data.DistributedSampler):
    def __init__(self, *args, **kwargs):
        super(MultiprocessingDistributedSampler, self).__init__(*args, **kwargs)
        self.num_repeats = 1
    
    def __len__(self):
        return len(self.dataset) * self.num_repeats

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        indices = indices * self.num_repeats
        return iter(indices)

class RepeatedSampler(torch.utils.data.Sampler):
    """
    Creates a sampler that repeatedly sample from an image
    """
    def __init__(self, dataset):
        super().__init__()

        self.dataset = dataset
        # self.num_samples = self.get_num_samples()
        self.num_samples = 10

    def get_num_samples(self):
        num_samples = []
        for metadata in self.dataset.metadata():
            item = metadata.item()
            sizeX, sizeY = item["msr-metadata"]["SizeX"], item["msr-metadata"]["SizeY"]
            num_samples.append(max(1, round((sizeX * sizeY) / (224 * 224))))
        return num_samples

    def __len__(self) -> int: 
        # return sum(self.num_samples)
        return len(self.dataset) * self.num_samples

    def __iter__(self):
        samples_per_image = []
        for i in range(len(self.dataset)):
            samples_per_image.extend([i] * self.num_samples)
        random.shuffle(samples_per_image)
        return iter(samples_per_image)

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
            use_cache=self.cfg.use_cache, 
            cache_system=cache_system, 
            max_cache_size=self.cfg.max_cache_size,
            world_size = self.world_size, rank = self.rank,
            return_metadata=self.cfg.return_metadata,
            **self.kwargs
        )
        
    def train_dataloader(self):
        # sampler = RepeatedSampler(self.dataset)
        
        if self.cfg.num_workers is None:
            num_workers = os.environ.get("SLURM_CPUS_PER_TASK", None)
            if num_workers is None:
                num_workers = os.cpu_count()
        else:
            num_workers = self.cfg.num_workers
        
        print("===============================")
        print("Num Workers: ", num_workers)
        print("===============================")

        sampler = MultiprocessingDistributedSampler(self.dataset, shuffle=self.cfg.shuffle)
        loader = torch.utils.data.DataLoader(
            self.dataset, 
            batch_size = self.cfg.batch_size,
            sampler = sampler,
            num_workers=int(num_workers),
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )
        return loader