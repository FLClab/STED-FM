
from torch.utils.data import Dataset
from dataclasses import dataclass

from .mRNAs import get_dataset as get_mRNAs_dataset

DATASETS = {
    "mRNAs-3b" : get_mRNAs_dataset,
    "mRNAs-3d" : get_mRNAs_dataset,
    "mRNAs-3f" : get_mRNAs_dataset
}

def get_dataset(name: str, cfg: dataclass, **kwargs) -> Dataset:
    if not name in DATASETS:
        raise NotImplementedError(f"`{name}` is not a valid option.")
    return DATASETS[name](name=name, cfg=cfg, **kwargs)