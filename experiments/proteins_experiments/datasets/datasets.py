import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import h5py
from torchvision import transforms
import numpy as np

class ProteinDataset(Dataset):
    def __init__(
            self, 
            h5file: str, 
            class_ids: List[int], 
            class_type: str, 
            n_channels: int = 1,
            indices: List[int] = None) -> None:
        self.h5file = h5file 
        self.class_ids = class_ids
        self.class_type = class_type
        self.n_channels = n_channels
        self.indices = indices

        if self.indices is None:
            with h5py.File(h5file, "r") as hf:
                self.dataset_size = int(hf["protein"].size)
        else:
            self.dataset_size = len(self.indices)

    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # If we operate from a predetermined list of indices, 
        # we need to convert the input index to the actual image index to be found in the hdf5
        if self.indices is not None:
            idx = self.indices[idx]

        with h5py.File(self.h5file, "r") as hf:
            img = hf["images"][idx]
            protein = hf["protein"][idx]
            if protein > 1:
                protein = protein - 1 # Because we removed the NKCC2 (label = 2) protein from our dataset
            condition = hf["condition"][idx]
            if self.n_channels == 3:
                img = np.tile(img[np.newaxis], (3, 1, 1))
                img = np.moveaxis(img, 0, -1)
                img = transforms.ToTensor()(img)
                img = transforms.Normalize(mean=[0.0695771782959453, 0.0695771782959453, 0.0695771782959453], std=[0.12546228631005282, 0.12546228631005282, 0.12546228631005282])(img)
            else:
                img = transforms.ToTensor()(img)
            return (img, protein, condition)
        
class TarFLCDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, x):
        pass
