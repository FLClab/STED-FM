import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import h5py
from torchvision import transforms

class ProteinDataset(Dataset):
    def __init__(self, h5file: str, class_ids: List[int], class_type: str) -> None:
        self.h5file = h5file 
        self.class_ids = class_ids
        self.class_type = class_type
        with h5py.File(h5file, "r") as hf:
            self.dataset_size = int(hf["protein"].size)

    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        with h5py.File(self.h5file, "r") as hf:
            img = hf["images"][idx]
            protein = hf["protein"][idx]
            condition = hf["condition"][idx]
            img = transforms.ToTensor()(img)
            return (img, protein, condition)
