import os
import glob
import re
import random
import numpy as np
import torch
import copy
from torch.utils.data import Dataset
from typing import List
from torchvision import transforms

class ProteinActivityDataset(Dataset):
    """
    "Block": 0,
    "0MgGlyBic": 1,
    "GluGly": 2,
    "48hTTX": 3,
    """
    def __init__(
        self,
        h5file: str,    
        transform: Callable = None,
        n_channels: int = 1,
        num_samples: int = None,
        num_classes: int = 2,
        protein_id: int = 3,
        balance: bool = True,
        keepclasses: List = [0, 1]
    ) -> None:
        self.h5file = h5file 
        self.transform = transform
        self.n_channels = n_channels
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.protein_id = protein_id
        self.balance = balance
        self.keepclasses = keepclasses

        with h5py.File(h5file, "r") as handle:
            images = handle["images"][()] 
            conditions = handle["conditions"][()] 
            proteins = handle["proteins"][()] 

        protein_mask = np.where(proteins == protein_id)
        images = images[protein_mask]
        conditions = conditions[protein_mask]
        proteins = proteins[protein_mask]
        class_mask = np.isin(conditions, self.keepclasses)
        images = images[class_mask]
        conditions = conditions[class_mask]
        proteins = proteins[class_mask]

        self.classes = {
            0: "Block",
            1: "0MgGlyBic",
            2: "GluGly",
            3: "48hTTX",
        }
        self.images = images
        self.conditions = conditions
        self.proteins = proteins

        if balance:
            self.__balance_classes()
        self.dataset_size = self.images.shape[0]


    def __balance_classes(self) -> None:
        uniques, counts = np.unique(self.conditions, return_counts=True)
        minority_count, minority_class = np.min(counts), np.argmin(counts)
        indices = []
        if self.num_samples is not None:
            minority_count = self.num_samples
        for unique in uniques:
            ids = np.where(self.conditions == unique)[0]
            ids = np.random.choice(ids, size=minority_count)
            indices.extend(ids)
        indices = np.sort(indices)
        self.images = self.images[indices]
        self.conditions = self.conditions[indices]
        self.proteins = self.proteins[indices]

    def __len__(self) -> int:
        return self.dataset_size 

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, protein, label = self.images[idx], self.proteins[idx], self.conditions[idx] 
        if self.n_channels == 3:
            img = np.tile(img[np.newaxis, :], (3, 1, 1))
            img = torch.tensor(img, dtype=torch.float32)
            img = transforms.Normalize(mean=[0.014, 0.014, 0.014], std=[0.12546228631005282, 0.12546228631005282, 0.12546228631005282])(img)
        else:
            img = torch.tensor(img[np.newaxis, :], dtype=torch.float32)
        img = self.transform(img) if self.transform is not None else img 
        return img, {"label": label, "protein": protein}



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
            classes: List = ['actin', 'tubulin', 'CaMKII', 'PSD95'],
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
                    labels.append(0)
                if quality_score < self.low_score_threshold:
                    filtered_files.append(file)
                    labels.append(1)
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
        return img, {"label" : label, "dataset-idx" : dataset_idx, "score" : quality_score}

    def __repr__(self):
        out = "\n"
        for key, values in self.samples.items():
            out += f"{key} - {len(values)}\n"        
        return "Dataset(optim) -- length: {}".format(len(self)) + out

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
