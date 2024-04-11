import tarfile 
import numpy as np
import io
import torch
from typing import Any, List, Tuple
from torch.utils.data import Dataset, get_worker_info
from tqdm import tqdm
from torchvision import transforms
import h5py

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

class CTCDataset(Dataset):
    def __init__(
            self,
            h5file: str,
            n_channels: int = 1,
            transform: Any = None,
    ) -> None:
        self.h5file = h5file
        self.n_channels = n_channels
        self.transform = transform
        with h5py.File(h5file, "r") as hf:
            self.dataset_size = int(hf["protein"].size)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx: int) -> torch.Tensor:
        with h5py.File(self.h5file, "r") as hf:
            img = hf["images"][idx]
            protein = hf['protein'][idx]
            condition = hf['condition'][idx]
            if self.transform is not None:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
            return img


class TarFLCDataset(Dataset):
    def __init__(
        self,
        tar_path: str, 
        use_cache: bool = False,
        max_cache_size: int = 32e9,
        image_channels: int = 1,
        transform: Any = None
    ) -> None:
        self.__cache = {}
        self.tar_path = tar_path
        self.image_channels = image_channels
        self.transform = transform
        self.max_cache_size = max_cache_size
        self.cache_size = 0
        self.transform = transform
        
        worker = get_worker_info()
        worker = worker.id if worker else None
        self.tar_obj = {worker: tarfile.open(self.tar_path, "r")}
        
        # store headers of all files and folders by name
        self.members = list(sorted(self.tar_obj[worker].getmembers(), key=lambda m: m.name))
        
        if use_cache and max_cache_size > 0:
            self.__fill_cache
    
    def __get_item_from_tar(self, member: tarfile.TarInfo):
        """
        Ensures a unique file handle per worker in a multiprocessing setting
        """
        worker = get_worker_info()
        worker = worker.id if worker else None
        if worker not in self.tar_obj:
            self.tar_obj[worker] = tarfile.open(self.tar_path, "r")
        buffer = io.BytesIO()
        buffer.write(self.tar_obj[worker].extractfile(member).read())
        buffer.seek(0)
        data = np.load(buffer, allow_pickle=True)
        return data
    
    def __getsizeof(self, obj: Any):
        if isinstance(obj, dict):
            return sum([self.__getsizeof(o) for o in obj.values()])
        elif isinstance(obj, (list, tuple)):
            return sum([self.__getsizeof(o) for o in obj])
        elif isinstance(obj, str):
            return len(str)
        else:
            return obj.size * obj.dtype.itemsize
    
    def __fill_cache(self):
        indices = np.arange(0, len(self.members), 1)
        np.random.shuffle(indices)
        print("Filling up the cache...")
        pbar = tqdm(indices, total=indices.shape[0])
        for i in pbar:
            if self.size >= self.max_cache_size:
                break
            data = self.__get_item_from_tar(self.members[i])
            self.__cache[i] = data
            self.size += self.__getsizeof(data)
            pbar.set_description(f"Cache size --> {self.size}")
    
    def __len__(self):
        return len(self.members)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx in self.__cache:
            data = self.__cache[idx]
        else:
            data = self.__get_item_from_tar(self.members[idx])
        img = data["image"]
        metadata = data["metadata"]
        img = img / 255.
        # img = torch.tensor(img, dtype=torch.float32)
        if self.transform is not None:
            print(f"In transform: {type(img)}")
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img).type(torch.FloatTensor)
        return img
    
    def __del__(self):
        for o in self.tar_obj.values():
            o.close()
    
    def __getstate__(self):
        state = dict(self.__dict__)
        state["tar_obj"] = {}
        return state