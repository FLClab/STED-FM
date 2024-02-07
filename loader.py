
import os 
import json
import tarfile
import numpy 
import io
from torch.utils.data import Dataset
import torch
from PIL import Image
from typing import Any

class TarFLCDataset(Dataset):
    def __init__(self, tar_path: str, use_cache: bool = False, max_cache_size: int = 16e9, image_channels: int = 3, transform: Any = None) -> None:
        self.__cache = {}
        self.tar_path = tar_path
        self.image_channels = image_channels
        self.transform = transform
        with tarfile.open(tar_path) as archive:
            self.members = archive.getmembers()

        if use_cache:
            self.__fill_cache()

    def __get_item_from_tar(self, fname: str):
        with tarfile.open(self.tar_path, "r") as archive:
            buffer = io.BytesIO()
            buffer.write(archive.extractfile(fname).read())
            buffer.seek(0)
            data = numpy.load(buffer, allow_pickle=True)
            # TODO
            # data will be a npz with image and metadata
            # We should decide if we return the full npz file here or only select elements
            # Returnning everything for now
            return data

    def __fill_cache(self):
        pass

    def __len__(self):
        return len(self.members)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx in self.__cache.keys():
            data = self.__cache[idx]
        else:
            data = self.__get_item_from_tar(self.members[idx].name) # will have to be a link between the idx and the fname here
        
        img = data["image"] # assuming 'img' key
        metadata = data["metadata"]
        
        if self.transform is not None:
            img = self.transform(img)
        return img # and whatever other metadata we like
            
if __name__ == "__main__":

    path = "/home-local2/projects/FLCDataset/dataset.tar"
    dataset = TarFLCDataset(path)
    for i in range(10):
        print(dataset[i])
