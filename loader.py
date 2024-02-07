
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
        with tarfile.open(tar_path) as archive:
            self.length = sum(1 for member in archive if member.isreg())

        if use_cache:
            self.__fill_cache()

    def __get_item_from_tar(self, fname: str):
        with tarfile.open(self.tar_path, "r") as archive:
            buffer = io.BytesIO()
            buffer.write(archive.extractfile(fname).read())
            buffer.seek(0)
            data = numpy.load(buffer)
            # TODO
            # data will be a npz with image and metadata
            # We should decide if we return the full npz file here or only select elements
            # Returnning everything for now
            return data

    def __fill_cache(self):
        pass

    def __len__(self):
        return self.length
    

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx in self.__cache.keys():
            data = self.__cache[idx]
        else:
            data = self.__get_item_from_tar(fname="temp") # will have to be a link between the idx and the fname here
            img = data["img"] # assuming 'img' key
            label = data["label"]
            if self.image_channels == 3:
                img = Image.fromarray((img * 255).astype('uint8')).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img, # and whatever other metadata we like
            





with tarfile.open("test.tar", "w") as tf:
    buffer = io.BytesIO()
    numpy.save(buffer, numpy.random.rand(128, 128).astype(numpy.float32))
    buffer.seek(0)

    info = tarfile.TarInfo(name="image1")
    info.size = len(buffer.getbuffer())
    tf.addfile(tarinfo=info, fileobj=buffer)

with tarfile.open("test.tar", "r") as tf:
    print(tf.getnames())
    buffer = io.BytesIO()
    buffer.write(tf.extractfile("image1").read())
    buffer.seek(0)
    img = numpy.load(buffer)
    print(img.shape)