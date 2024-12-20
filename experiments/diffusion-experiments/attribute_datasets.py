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

def get_dataset(name: str, training: bool = False,**kwargs): 
    if name == "quality":
        if training:
            train_dataset = OptimQualityDataset(
                data_folder="/home/frbea320/scratch/evaluation-data/optim_train",
                high_score_threshold=kwargs.get("high_score_threshold", 0.70),
                low_score_threshold=kwargs.get("low_score_threshold", 0.70),
                **kwargs
            )
            valid_dataset = OptimQualityDataset(
                data_folder="/home/frbea320/scratch/evaluation-data/optim_valid",
                high_score_threshold=kwargs.get("high_score_threshold", 0.70),
                low_score_threshold=kwargs.get("low_score_threshold", 0.70),
                **kwargs
            )
            return train_dataset, valid_dataset
        else:
            return OptimQualityDataset(
                data_folder="/home/frbea320/scratch/evaluation-data/optim-data",
                high_score_threshold=kwargs.get("high_score_threshold", 0.70),
                low_score_threshold=kwargs.get("low_score_threshold", 0.50),
                **kwargs
            )
    else:
        raise ValueError(f"Dataset {name} not implemented yet or invalid.")
