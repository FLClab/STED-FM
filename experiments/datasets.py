import tarfile 
import numpy as np
import io
import torch
from typing import Any, List, Tuple
from torch.utils.data import Dataset, get_worker_info
from tqdm import tqdm
from torchvision import transforms
import h5py
import random
import os 
import glob
import re
import tifffile
from skimage import filters

class CreateFactinRingsFibersDataset(Dataset):
    def __init__(self, data_folder : str, transform : Any, classes : list, requires_3_channels : bool=False):
    
        self.data_folder = data_folder
        self.transform = transform
        self.classes = classes
        self.requires_3_channels = requires_3_channels
        
        self.samples = {}
        for class_name in self.classes:
            files = glob.glob(os.path.join(data_folder, f"**/*{class_name}*"), recursive=True)
#             for file in files:
#                 image_id, channel = os.path.basename(file).split("_")[-1].split(".")[0].split("-")
                
            self.samples[class_name] = files
    
    def __len__(self):
        return sum(len(value) for value in self.samples.values())
    
    def __getitem__(self, idx : int):
        
        class_name = None
        class_index = None
        index = None
        
        dataset_idx = idx

        for i, class_name in enumerate(self.classes):
            if idx < len(self.samples[class_name]):
                file_name = self.samples[class_name][idx]
                class_folder = class_name
                class_index = i
                index = idx
                break
            else:
                idx -= len(self.samples[class_name])

        path = file_name
        label = class_index
        
        image_folder = "/home-local/Multilabel-Proteins-Actin/Segmentation"
        image_id, channel = os.path.basename(path).split("_")[-1].split(".")[0].split("-")
        path = os.path.join(image_folder, f"{image_id}_image.tif")
        image = tifffile.imread(path)[int(channel)]
        
        m, M = image.min(), image.max()
        image = (image - m) / (M - m)
        if self.requires_3_channels:
            image = numpy.tile(image[numpy.newaxis], (3, 1, 1))
        else:
            image = image[numpy.newaxis]
        image = torch.tensor(image, dtype=torch.float32)        
        
        if self.transform:
            image = self.transform(image)
        
        return image, {"label" : label, "dataset-idx" : dataset_idx}
    
    def __repr__(self):
        out = "\n"
        for key, values in self.samples.items():
            out += f"{key} - {len(values)}\n"        
        return "Dataset(F-actin) -- length: {}".format(len(self)) + out

class CreateFActinBlockGluGlyDataset(Dataset):
    def __init__(self, data_folder : str, transform : Any, classes : list, requires_3_channels : bool=False):
    
        self.data_folder = data_folder
        self.transform = transform
        self.classes = classes
        self.requires_3_channels = requires_3_channels
        
        self.samples = {}
        for class_name in self.classes:
            files = glob.glob(os.path.join(data_folder, f"{class_name}/*merged.tif"), recursive=True)
            self.samples[class_name] = self.get_valid_indices(files)
        
    def get_valid_indices(self, files: list, crop_size=224):
        out = []
        for file in files:
            image = tifffile.imread(file)
            # Dendrite foreground
            threshold = filters.threshold_otsu(image[2])
            foreground = image[2] > threshold
            for j in range(0, image.shape[-2], crop_size):
                for i in range(0, image.shape[-1], crop_size):
                    slc = (
                        slice(j, j + crop_size) if j + crop_size < image.shape[-2] else slice(image.shape[-2] - crop_size, image.shape[-2]),
                        slice(i,  i + crop_size) if i + crop_size < image.shape[-1] else slice(image.shape[-1] - crop_size, image.shape[-1]),
                    )                    
                    crop = foreground[slc]
                    if crop.sum() > 0.1 * crop.size:
                        out.append({
                            "path" : file,
                            "slc" : slc
                        })
        return out
    
    def __len__(self):
        return sum(len(value) for value in self.samples.values())
    
    def __getitem__(self, idx : int):
        
        class_name = None
        class_index = None
        index = None
        
        dataset_idx = idx

        for i, class_name in enumerate(self.classes):
            if idx < len(self.samples[class_name]):
                info = self.samples[class_name][idx]
                class_folder = class_name
                class_index = i
                index = idx
                break
            else:
                idx -= len(self.samples[class_name])

        path = info["path"]
        label = class_index
        
        image = tifffile.imread(path)[0]
        image = image[info["slc"]]

        m, M = image.min(), image.max()
        image = (image - m) / (M - m)
        if self.requires_3_channels:
            image = np.tile(image[np.newaxis], (3, 1, 1))
        else:
            image = image[np.newaxis]
        image = torch.tensor(image, dtype=torch.float32)        
        
        if self.transform:
            image = self.transform(image)
        
        return image, {"label" : label, "dataset-idx" : dataset_idx}
    
    def __repr__(self):
        out = "\n"
        for key, values in self.samples.items():
            out += f"{key} - {len(values)}\n"
        return "Dataset(F-actin) -- length: {}".format(len(self)) + out


class OptimDataset(Dataset):
    """
    Dataset class for loading and processing image data from different classes.
        
    Args:
        data_folder (str): path to the root data folder containing subfolders for each class.
        num_samples (dict or None): number of samples to randomly select from each class.
        transform (callable, optional): transformation to apply on each image.
        apply_filter (bool): choose to filter files based on quality score or not.
        classes (list): list of class names present in the dataset.
    """
    def __init__(
            self,
            data_folder: str,
            num_samples = None,
            transform = None,
            apply_filter: bool = False, 
            classes: List = ['actin', 'tubulin', 'CaMKII', 'PSD95'],
            n_channels: int = 1,
    ) -> None:
        self.data_folder = data_folder
        self.num_samples = num_samples
        self.transform = transform
        self.apply_filter = apply_filter
        self.classes = classes 
        self.n_channels = n_channels
        self.class_files = {}
        self.samples = {}

        random.seed(42)
        np.random.seed(42)
        for class_name in classes:
            class_folder = os.path.join(data_folder, class_name)
            self.class_files[class_name] = self._filter_files(class_folder)
            self.samples[class_name] = self._get_sampled_files(self.class_files[class_name], self.num_samples.get(class_name))

    def _filter_files(self, class_folder):
        SCORE = 0.70
        files = glob.glob(os.path.join(class_folder, "**/*.npz"), recursive=True)
        filtered_files = []
        for file in files:
            match = re.search(r"-(\d+\.\d+)\.npz", file)
            if match:
                quality_score = float(match.group(1))
                if not self.apply_filter or quality_score >= SCORE:
                    filtered_files.append(file)
        return filtered_files

    def _get_sampled_files(self, files_list, num_sample):
        if num_sample is not None:
            return random.sample(files_list, num_sample)
        else:
            return files_list

    def __len__(self):
        total_length = sum(len(self.samples[class_name]) for class_name in self.classes)
        return total_length

    def __getitem__(self, idx: int) -> torch.Tensor:
        class_name = None
        class_index = None
        index = None
        dataset_idx = idx 
        for i, class_name in enumerate(self.classes):
            if idx < len(self.samples[class_name]):
                file_name = self.samples[class_name][idx]
                class_folder = class_name
                class_index = i
                index = idx
                break
            else:
                idx -= len(self.samples[class_name])

        path = file_name
        label = class_index 
        match = re.search(r"-(\d+\.\d+)\.npz", path)
        if match:
            quality_score = float(match.group(1))

        data = np.load(path)
        image = data['arr_0']
        
        m, M = np.quantile(image, [0.01, 0.995])
        m, M = image.min(), image.max()
        image = (image - m) / (M - m)
        if self.requires_3_channels:
            image = np.tile(image[np.newaxis], (3, 1, 1))
        else:
            image = image[np.newaxis]
        image = torch.tensor(image, dtype=torch.float32)        
        
        if self.transform:
            image = self.transform(image)
        
        return image, {"label" : label, "dataset-idx" : dataset_idx, "score" : quality_score}

    def __repr__(self):
        out = "\n"
        for key, values in self.samples.items():
            out += f"{key} - {len(values)}\n"        
        return "Dataset(optim) -- length: {}".format(len(self)) + out

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
            return img, {"protein": protein, "condition": condition}

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
            return img, {"protein": protein, "condition": condition}


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