
from torch.utils.data import Dataset
from dataclasses import dataclass

from .jc_factin import get_dataset as get_jc_factin_dataset

DATASETS = {
    "jc-factin-tau" : get_jc_factin_dataset,
}

def get_dataset(name: str, cfg: dataclass, **kwargs) -> Dataset:
    if not name in DATASETS:
        raise NotImplementedError(f"`{name}` is not a valid option.")
    return DATASETS[name](name=name, cfg=cfg, **kwargs)