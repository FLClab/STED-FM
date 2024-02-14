
import tarfile
import numpy 
import io
import torch

from typing import Any
from tqdm.auto import tqdm
from torch.utils.data import Dataset, get_worker_info

class TarFLCDataset(Dataset):
    def __init__(self, tar_path: str, use_cache: bool = False, max_cache_size: int = 16e9, image_channels: int = 3, transform: Any = None, cache_system=None) -> None:
        self.__cache = {}
        self.__max_cache_size = max_cache_size
        self.tar_path = tar_path
        self.image_channels = image_channels
        self.transform = transform

        worker = get_worker_info()
        worker = worker.id if worker else None
        self.tar_obj = {worker: tarfile.open(self.tar_path, "r")}

        # store headers of all files and folders by name
        self.members = list(sorted(self.tar_obj[worker].getmembers(), key=lambda m: m.name))

        if use_cache:
            self.__cache_size = 0
            if not cache_system is None:
                self.__cache = cache_system
            self.__fill_cache()

    def __get_item_from_tar(self, member: tarfile.TarInfo):

        # ensure a unique file handle per worker, in multiprocessing settings
        worker = get_worker_info()
        worker = worker.id if worker else None
        if worker not in self.tar_obj:
            self.tar_obj[worker] = tarfile.open(self.tar_path, "r")

        # Loads file from TarFile stored as numpy array
        buffer = io.BytesIO()
        buffer.write(self.tar_obj[worker].extractfile(member).read())
        buffer.seek(0)
        data = numpy.load(buffer, allow_pickle=True)

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
        indices = numpy.arange(0, len(self.members), 1)
        numpy.random.shuffle(indices)
        print("Filling up the cache...")
        pbar = tqdm(indices, total=indices.shape[0])
        for idx in pbar:
            if self.__cache_size >= self.__max_cache_size:
                break
            data = self.__get_item_from_tar(self.members[idx])
            data = {key : values for key, values in data.items()}
            self.__cache[idx] = data
            self.__cache_size += self.__getsizeof(data)
            pbar.set_description(f"Cache size --> {self.__cache_size * 1e-9:0.2f}G")

    def __len__(self):
        return len(self.members)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx in self.__cache:
            data = self.__cache[idx]
        else:
            data = self.__get_item_from_tar(self.members[idx])
        
        img = data["image"] # assuming 'img' key
        metadata = data["metadata"]
        if img.size != 224 * 224:
            print(img.shape)
            print(metadata)
        
        img = img / 255.
        img = img[numpy.newaxis]
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