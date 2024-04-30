from torch.utils.data import Dataset
from dataclasses import dataclass 
from .synprot import get_dataset as get_synaptic_protein_dataset

DATASETS = {
    "synaptic-denoising": get_synaptic_protein_dataset
}

def get_dataset(name: str, cfg: dataclass, **kwargs) -> Dataset:
    for key, value in kwargs.items():
        print(f"{key}: {value}")
    if not name in DATASETS:
        raise NotImplementedError(f"{name} is not a valid dataset.")
    return DATASETS[name](cfg=cfg, **kwargs)