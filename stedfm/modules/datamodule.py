
import torch
import random
import os

from torch.utils.data import default_collate
from multiprocessing import Manager
from lightning.pytorch.core import LightningDataModule

import torch.distributed
from stedfm.datasets import get_dataset

class MultiprocessingDistributedSampler(torch.utils.data.DistributedSampler):
    """
    A custom Distributed Sampler that works with multiprocessing.
    """
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
        self.args = args
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
        if self.dataset_name == "Hybrid":
            hpa_path = self.args.hpa_path 
            sim_path = self.args.sim_path 
            sted_path = self.args.sted_path 
            self.dataset = get_dataset(
                self.dataset_name, "",
                hpa_path=hpa_path,
                sim_path=sim_path,
                sted_path=sted_path,
                use_cache=self.cfg.datamodule.use_cache,
                cache_system=cache_system,
                max_cache_size=self.cfg.datamodule.max_cache_size,
                world_size = self.world_size, rank = self.rank,
                return_metadata=self.cfg.datamodule.return_metadata,
                **self.kwargs
            )
        else:
            self.dataset = get_dataset(
                self.dataset_name, self.dataset_path, 
                use_cache=self.cfg.datamodule.use_cache, 
                cache_system=cache_system, 
                max_cache_size=self.cfg.datamodule.max_cache_size,
                world_size = self.world_size, rank = self.rank,
                return_metadata=self.cfg.datamodule.return_metadata,
                **self.kwargs
            )
        
    def train_dataloader(self):
        # sampler = RepeatedSampler(self.dataset)
        
        if self.cfg.datamodule.num_workers is None:
            num_workers = os.environ.get("SLURM_CPUS_PER_TASK", None)
            if num_workers is None:
                num_workers = os.cpu_count()
        else:
            num_workers = self.cfg.datamodule.num_workers
        
        if self.trainer.current_epoch == 0:
            print("===============================")
            print("Num Workers: ", num_workers)
            print("===============================")

        sampler = MultiprocessingDistributedSampler(self.dataset, shuffle=self.cfg.datamodule.shuffle)
        if self.dataset_name in ["MCSTED"]:
            collate_fn = multichannel_collate_fn
        else:
            collate_fn = default_collate
        loader = torch.utils.data.DataLoader(
            self.dataset, 
            batch_size = self.cfg.batch_size,
            sampler = sampler,
            num_workers=int(num_workers),
            pin_memory=True if self.trainer.reload_dataloaders_every_n_epochs == 0 else False, # Pin memory only if we are not reloading dataloaders every epoch (since reloading dataloaders every epoch with pin_memory=True seems to cause issues with multiprocessing)
            persistent_workers=True,
            drop_last=True,
            collate_fn=collate_fn
        )
        return loader
    
def multichannel_collate_fn(batch):
    elem = batch[0]
    if isinstance(elem, tuple) and isinstance(elem[0], torch.Tensor):
        # If the element is a tuple, we assume it is of the form (image, metadata) and we collate the images and return the metadata as a list
        batched_per_num_channels = {}
        for item in batch:
            num_channels = item[0].shape[0]
            if num_channels not in batched_per_num_channels:
                batched_per_num_channels[num_channels] = []
            batched_per_num_channels[num_channels].append(item)
        batch = []
        for num_channels, items in batched_per_num_channels.items():
            images = torch.stack([item[0] for item in items], dim=0)
            metadata = [item[1] for item in items]
            batch.append((images, metadata))
        return batch
    elif isinstance(elem, torch.Tensor):
        # Since the images can have different sizes, we cannot stack them into a single tensor. We return a list of tensors instead.
        # Optionally, we could merge images with the same size into a single tensor
        batched_per_num_channels = {}
        for item in batch:
            num_channels = item.shape[0]
            if num_channels not in batched_per_num_channels:
                batched_per_num_channels[num_channels] = []
            batched_per_num_channels[num_channels].append(item)
        batch = []
        for num_channels, items in batched_per_num_channels.items():
            images = torch.stack([item for item in items], dim=0)
            batch.append(images)
        return batch
    else:
        return default_collate(batch)
