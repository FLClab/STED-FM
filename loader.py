
import os 
import time
import json
import tarfile
import numpy 
import io
from torch.utils.data import Dataset, DataLoader, get_worker_info
import torch
from PIL import Image
from typing import Any

from tqdm.auto import tqdm

class TarFLCDataset(Dataset):
    def __init__(self, tar_path: str, use_cache: bool = False, max_cache_size: int = 16e9, image_channels: int = 3, transform: Any = None) -> None:
        self.__cache = {}
        self.tar_path = tar_path
        self.image_channels = image_channels
        self.transform = transform

        worker = get_worker_info()
        worker = worker.id if worker else None
        self.tar_obj = {worker: tarfile.open(self.tar_path, "r")}

        # store headers of all files and folders by name
        self.members = list(sorted(self.tar_obj[worker].getmembers(), key=lambda m: m.name))
        self.members_by_name = {m.name: m for m in self.members}       

        if use_cache:
            self.__fill_cache()

    def __get_item_from_tar(self, member: tarfile.TarInfo):

        # ensure a unique file handle per worker, in multiprocessing settings
        worker = get_worker_info()
        worker = worker.id if worker else None
        if worker not in self.tar_obj:
            self.tar_obj[worker] = tarfile.open(self.tar_path, "r")

        buffer = io.BytesIO()
        buffer.write(self.tar_obj[worker].extractfile(self.members_by_name[member.name]).read())
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
            data = self.__get_item_from_tar(self.members[idx]) # will have to be a link between the idx and the fname here
        
        img = data["image"] # assuming 'img' key
        metadata = data["metadata"]
        
        img = img / 255.
        img = torch.tensor(img, dtype=torch.float32)

        if self.transform is not None:
            img = self.transform(img)
        return img # and whatever other metadata we like
    
    def __del__(self):
        """
        Close the TarFile file handles on exit.
        """
        for o in self.tar_obj.values():
            o.close()
            
    def __getstate__(self):
        """
        Serialize without the TarFile references, for multiprocessing compatibility.
        """
        state = dict(self.__dict__)
        state['tar_obj'] = {}
        return state

if __name__ == "__main__":

    path = "/home-local2/projects/FLCDataset/dataset.tar"
    dataset = TarFLCDataset(path)
    dataloader = DataLoader(dataset=dataset, batch_size=128, num_workers=4)
    import time 
    start = time.time()
    for i, X in enumerate(tqdm(dataloader)):
        X
    print(time.time() - start)