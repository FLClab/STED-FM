
from torch.utils.data import Dataset
from dataclasses import dataclass

from .actin import get_dataset as get_actin_dataset
from .fp import get_dataset as get_fp_dataset
from .synprot import get_dataset as get_synaptic_protein_dataset
from .lioness import get_dataset as get_lioness_dataset
from .zooniverse import get_dataset as get_zooniverse_dataset
from .mitochondria import get_dataset as get_mitochondria_dataset

DATASETS = {
    "factin" : get_actin_dataset,
    "footprocess" : get_fp_dataset,
    'synaptic-protein-segmentation': get_synaptic_protein_dataset,
    "lioness" : get_lioness_dataset,
    "zooniverse": get_zooniverse_dataset,
    "synaptic-semantic-segmentation" : get_synaptic_protein_dataset,
    "perforated-segmentation" : get_synaptic_protein_dataset,
    "multidomain-detection" : get_synaptic_protein_dataset,
    "lioness" : get_lioness_dataset,
    "mitochondria" : get_mitochondria_dataset,
}

def get_dataset(name: str, cfg: dataclass, **kwargs) -> Dataset:
    if not name in DATASETS:
        raise NotImplementedError(f"`{name}` is not a valid option.")
    return DATASETS[name](name=name, cfg=cfg, **kwargs)