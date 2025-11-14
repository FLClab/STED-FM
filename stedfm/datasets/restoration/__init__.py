
from torch.utils.data import Dataset
from dataclasses import dataclass

from .lqhq import get_dataset as get_lqhq_dataset
from .lioness import get_dataset as get_lioness_dataset
from .ov_lqhq import get_dataset as get_ov_lqhq_dataset
from .unet_rcan import get_dataset as get_unet_rcan_dataset
from .rcan3d_sr import get_dataset as get_3drcan_dataset
from .dendritic_factin import get_dataset as get_dendritic_factin_dataset

DATASETS = {
    "jmb-lqhq" : get_lqhq_dataset,
    "kt-lqhq" : get_lqhq_dataset,
    "kt-lqhq-vgat" : get_lqhq_dataset,
    "kt-lqhq-gephyrin" : get_lqhq_dataset,
    "lioness-lqhq": get_lioness_dataset,
    "ov-lqhq-live-mito": get_ov_lqhq_dataset,
    "ov-lqhq-mt": get_ov_lqhq_dataset,
    "3d-rcan-mt": get_3drcan_dataset,
    "3d-rcan-npc": get_3drcan_dataset,
    "unet-rcan-mt": get_unet_rcan_dataset,
    "unet-rcan-hist": get_unet_rcan_dataset,
    "unet-rcan-tub": get_unet_rcan_dataset,
    "dendritic-factin": get_dendritic_factin_dataset,
}

def get_dataset(name: str, cfg: dataclass, **kwargs) -> Dataset:
    if not name in DATASETS:
        raise NotImplementedError(f"`{name}` is not a valid option.")
    return DATASETS[name](name=name, cfg=cfg, **kwargs)