
from torch.utils.data import Dataset
from dataclasses import dataclass

from .actin import get_dataset as get_actin_dataset


DATASETS = {
    "factin" : get_actin_dataset,
}

def get_dataset(name: str, cfg: dataclass, **kwargs) -> Dataset:
    if not name in DATASETS:
        raise NotImplementedError(f"`{name}` is not a valid option.")
    return DATASETS[name](name=name, cfg=cfg, **kwargs)
