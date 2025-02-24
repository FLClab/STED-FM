import os
import glob
import re
import random
import numpy as np
from sympy import igcd
import torch
import copy
from torch.utils.data import Dataset
from typing import List, Optional, Callable, Tuple
from torchvision import transforms
import h5py
import tarfile
import io
from tqdm import tqdm

class LowHighResolutionDataset(Dataset):
    def __init__(
            self,
            h5path: str,
            transform: Optional[Callable] = None,
            n_channels: int = 1,
            num_samples: int = None,
            num_classes: int = 2,
            classes: List[str] = ["low", "high"],
    ) -> None:
        with h5py.File(h5path, "r") as handle:
            high_images = handle["high/rabBassoon STAR635P"][()]
            high_labels = [1] * len(high_images)
            low_images = handle["low/rabBassoon STAR635P"][()]
            low_labels = [0] * len(low_images)
        self.images = np.concatenate([high_images, low_images], axis=0)
        self.labels = np.concatenate([high_labels, low_labels], axis=0)

        indices = np.arange(self.images.shape[0])
        np.random.shuffle(indices)
        self.images = self.images[indices]
        self.labels = self.labels[indices]

        self.transform = transform
        self.n_channels = n_channels
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.classes = classes
        self.dataset_size = self.images.shape[0]

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, label = self.images[idx], self.labels[idx]
        if self.n_channels == 3:
            img = torch.tensor(img, dtype=torch.float32)
            img = img.repeat(3, 1, 1)
            img = transforms.Normalize(mean=[0.010903545655310154, 0.010903545655310154, 0.010903545655310154], std=[0.03640301525592804, 0.03640301525592804, 0.03640301525592804])(img)
        else:
            img = torch.tensor(img[np.newaxis, :], dtype=torch.float32)
        img = self.transform(img) if self.transform is not None else img
        return img, {"label": label}

class ProteinActivityDataset(Dataset):
    def __init__(
            self,
            tarpath: str, 
            transform: Callable = None,
            n_channels: int = 1,
            num_samples: int = None,
            num_classes: int = 4,
            classes: List[str] = ["Block", "0MgGlyBic", "GluGly", "48hTTX"],
            balance: bool = True
    ) -> None:
        self.tarpath = tarpath
        self.transform = transform
        self.n_channels = n_channels
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.classes = classes


        imgs = []
        masks = []
        conditions = []
        with tarfile.open(tarpath, "r") as handle:
            names = handle.getnames()
            for name in tqdm(names, desc="Processing dataset.."):
                if name.split("-")[0] not in self.classes:
                    continue
                buffer = io.BytesIO()
                buffer.write(handle.extractfile(name).read())
                buffer.seek(0)
                data = np.load(buffer, allow_pickle=True)
                data = {key : values for key, values in data.items()}
                imgs.append(data["image"])
                masks.append(data["mask"])
                metadata = data["metadata"].item()
                conditions.append(metadata["condition"])

        self.imgs = imgs
        self.masks = masks
        self.conditions = conditions
        self.labels = [self.classes.index(condition) for condition in self.conditions]
        if balance:
            indices = self.__balance_classes(conditions)
            self.imgs = [imgs[i] for i in indices]
            self.masks = [masks[i] for i in indices]
            self.conditions = [conditions[i] for i in indices]
            self.labels = [self.classes.index(condition) for condition in self.conditions]
            
        assert len(self.imgs) == len(self.masks) == len(self.labels) == len(self.conditions)

    def __balance_classes(self, conditions: List[str]) -> None:
        np.random.seed(42)
        conditions = np.array(conditions)
        uniques, counts = np.unique(conditions, return_counts=True)
        minority_count, minority_class = np.min(counts), np.argmin(counts)
        print(uniques, counts)
        if self.num_samples is not None:
            minority_count = self.num_samples
        indices = []
        for unique in uniques:
            ids = np.where(conditions == unique)[0]
            ids = np.random.choice(ids, size=minority_count, replace=False)
            indices.extend(ids)
        indices = np.sort(indices)
        return indices

    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        img, mask, label, condition = self.imgs[idx], self.masks[idx], self.labels[idx], self.conditions[idx]

        if self.n_channels == 3:
            img = np.tile(img[np.newaxis, :], (3, 1, 1))
            img = torch.tensor(img, dtype=torch.float32)
            # img = transforms.Normalize(mean=[0.0695771782959453, 0.0695771782959453, 0.0695771782959453], std=[0.12546228631005282, 0.12546228631005282, 0.12546228631005282])(img)
            img = transforms.Normalize(mean=[0.014, 0.014, 0.014], std=[0.03, 0.03, 0.03])(img)
        else:
            img = torch.tensor(img[np.newaxis, :], dtype=torch.float32)

        img = self.transform(img) if self.transform is not None else img

        return img, {"mask": mask, "label": label, "condition": condition, "dataset-idx": idx}

class OptimQualityDataset(Dataset):
    """
    Dataset class for loading and processing image data from different classes.
        
    Args:
        data_folder (str): path to the root data folder containing subfolders for each class.
        num_samples (dict or None): number of samples to randomly select from each class.
        transform (callable, optional): transformation to apply on each image.
        classes (list): list of class names present in the dataset.
        high_score_threshold (float): lower threshold for high quality images.
        low_score_threshold (float): upper threshold for low quality images.
    """
    def __init__(
            self,
            data_folder: str,
            num_samples = None,
            transform = None,
            classes: List = ['actin', 'tubulin', 'CaMKII_Neuron', 'PSD95_Neuron'],
            n_channels: int = 1,
            high_score_threshold: float = 0.70,
            low_score_threshold: float = 0.60,
            *args, **kwargs
    ) -> None:
        self.data_folder = data_folder
        self.num_samples = num_samples
        self.transform = transform
        self.classes = classes 
        self.num_classes = len(self.classes)
        self.n_channels = n_channels
        self.samples = {}
        self.num_classes = len(classes)
        self.high_score_threshold = high_score_threshold
        self.low_score_threshold = low_score_threshold
        self.labels = {}

        original_size = 0
        for i, class_name in enumerate(classes):
            class_folder = os.path.join(data_folder, class_name)
            self.samples[class_name], self.labels[class_name] = self._filter_files(class_folder)
            print(f"Samples in {class_name}: {len(self.samples[class_name])}")

    def _filter_files(self, class_folder):
        files = glob.glob(os.path.join(class_folder, "**/*.npz"), recursive=True)
        filtered_files = []
        labels = []
        for file in files:
            match = re.search(r"-(\d+\.\d+)\.npz", file)
            if match:
                quality_score = float(match.group(1))
                if quality_score >= self.high_score_threshold:
                    filtered_files.append(file)
                    labels.append(1)
                if quality_score < self.low_score_threshold:
                    filtered_files.append(file)
                    labels.append(0)
        return filtered_files, labels

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
                label = self.labels[class_name][idx]
                class_folder = class_name
                class_index = i
                index = idx
                break
            else:
                class_folder = class_name
                idx -= len(self.samples[class_name])

        path = file_name
        match = re.search(r"-(\d+\.\d+)\.npz", path)
        if match:
            quality_score = float(match.group(1))

        data = np.load(path)
        image = data['arr_0']
        
        m, M = np.quantile(image, [0.01, 0.995])
        m, M = image.min(), image.max()
        image = (image - m) / (M - m)
        # image = np.tile(image[np.newaxis], (self.n_channels, 1, 1))
        # image = torch.tensor(image, dtype=torch.float32)   
        if self.n_channels == 3:
                img = np.tile(image[np.newaxis], (3, 1, 1))
                img = np.moveaxis(img, 0, -1)
                img = transforms.ToTensor()(img)
                img = transforms.Normalize(mean=[0.0695771782959453, 0.0695771782959453, 0.0695771782959453], std=[0.12546228631005282, 0.12546228631005282, 0.12546228631005282])(img)
        else:
            img = transforms.ToTensor()(image)
        
        img = self.transform(img) if self.transform is not None else img
        
        # label = np.float64(label)
        return img, {"label" : label, "dataset-idx" : dataset_idx, "score" : quality_score, "protein": class_folder}

    def __repr__(self):
        out = "\n"
        for key, values in self.samples.items():
            out += f"{key} - {len(values)}\n"        
        return "Dataset(optim) -- length: {}".format(len(self)) + out
    
class TubulinActinDataset(Dataset):
    def __init__(
            self,
            data_folder: str,
            num_samples: int = None,
            transform: Optional[Callable] = None, 
            classes: List = ["tubulin", "actin"],
            n_channels: int = 1,
            min_quality_score: float = 0.70
    ) -> None:
        self.data_folder = data_folder 
        self.num_samples = num_samples 
        self.transform = transform 
        self.classes = classes 
        self.n_channels = n_channels 
        self.min_quality_score = min_quality_score 

        self.samples = {}
        self.labels = []
        
        original_size = 0
        for i, class_name in enumerate(classes):
            class_folder = os.path.join(data_folder, class_name)
            self.samples[class_name] = self._filter_files(class_folder)
            self.labels.extend([i] * len(self.samples[class_name]))
            original_size += len(self.samples[class_name])
        self.original_size = original_size

    def _filter_files(self, class_folder):
        files = glob.glob(os.path.join(class_folder, "**/*.npz"), recursive=True) 
        filtered_files = []
        for file in files:
            match = re.search(r"-(\d+\.\d+)\.npz", file)
            if match:
                quality_score = float(match.group(1))
                if quality_score >= self.min_quality_score:
                    filtered_files.append(file)
        return list(sorted(filtered_files))

    def __len__(self) -> int:
        return self.original_size  

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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

        if self.n_channels == 3:
            img = np.tile(image[np.newaxis], (3, 1, 1))
            img = np.moveaxis(img, 0, -1)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize(mean=[0.0695771782959453, 0.0695771782959453, 0.0695771782959453], std=[0.12546228631005282, 0.12546228631005282, 0.12546228631005282])(img)
        else:
            img = transforms.ToTensor()(image)
        
        img = self.transform(img) if self.transform is not None else img 
        return img, {"label": label, "dataset-idx": dataset_idx, "score": quality_score}

def get_dataset(name: str, training: bool = False, **kwargs): 
    if name == "quality":
        if training:
            train_dataset = OptimQualityDataset(
                data_folder="/home-local/Frederic/Datasets/evaluation-data/optim_train",
                high_score_threshold=kwargs.get("high_score_threshold", 0.70),
                low_score_threshold=kwargs.get("low_score_threshold", 0.70),
                **kwargs
            )
            valid_dataset = OptimQualityDataset(
                data_folder="/home-local/Frederic/Datasets/evaluation-data/optim_valid",
                high_score_threshold=kwargs.get("high_score_threshold", 0.70),
                low_score_threshold=kwargs.get("low_score_threshold", 0.70),
                **kwargs
            )
            return train_dataset, valid_dataset
        else:
            return OptimQualityDataset(
                data_folder="/home-local/Frederic/Datasets/evaluation-data/optim-data",
                high_score_threshold=kwargs.get("high_score_threshold", 0.70),
                low_score_threshold=kwargs.get("low_score_threshold", 0.50),
                **kwargs
            )
    else:
        raise ValueError(f"Dataset {name} not implemented yet or invalid.")
