
from torch.utils.data import Dataset
from dataclasses import dataclass

from .lqhq import get_dataset as get_lqhq_dataset
from .ov_lqhq import get_dataset as get_ov_lqhq_dataset
from .rcan3d_sr import get_dataset as get_3drcan_dataset

DATASETS = {
    "lqhq" : get_lqhq_dataset,
    "ov-lqhq-live-mito": get_ov_lqhq_dataset,
    "ov-lqhq-mt": get_ov_lqhq_dataset,
    "3d-rcan-mt": get_3drcan_dataset,
    "3d-rcan-npc": get_3drcan_dataset
}

def get_dataset(name: str, cfg: dataclass, **kwargs) -> Dataset:
    if not name in DATASETS:
        raise NotImplementedError(f"`{name}` is not a valid option.")
    return DATASETS[name](name=name, cfg=cfg, **kwargs)