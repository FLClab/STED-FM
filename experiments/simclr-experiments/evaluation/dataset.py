
import os
import random
import numpy
import glob
import re 
import torch
import tifffile

from typing import Any
from torch.utils.data import Dataset
from skimage import filters

def filter_files(class_folder, apply_filter=True):
    files = os.listdir(class_folder)
    filtered_files = []

    for file in files:
        match = re.search(r"-(\d+\.\d+)\.npz", file)
        if match:
            quality_score = float(match.group(1))
            if not apply_filter or quality_score >= 0.70:
                filtered_files.append(file)

    return filtered_files

class CreateOptimDataset(Dataset):
    """
    Dataset class for loading and processing image data from different classes.
        
    Args:
        data_folder (str): path to the root data folder containing subfolders for each class.
        num_samples (dict or None): number of samples to randomly select from each class.
        transform (callable, optional): transformation to apply on each image.
        apply_filter (bool): choose to filter files based on quality score or not.
        classes (list): list of class names present in the dataset.
    """
    def __init__(self, data_folder, num_samples=None, transform=None, apply_filter=False, classes=['actin','tubulin','CaMKII','PSD95'], requires_3_channels=False):
        self.data_folder = data_folder
        self.num_samples = num_samples
        self.transform = transform
        self.apply_filter = apply_filter
        self.classes = classes
        self.requires_3_channels = requires_3_channels

        self.class_files = {}
        self.samples = {}

        random.seed(20)
        numpy.random.seed(20)

        # Loop through each class and process files
        for class_name in classes:
            class_folder = os.path.join(data_folder, class_name)
            # Filter files
            self.class_files[class_name] = self.filter_files(class_folder)
             # Randomly sample files based on num_samples
            self.samples[class_name] = self.get_sampled_files(self.class_files[class_name], self.num_samples.get(class_name))

    def filter_files(self, class_folder):
        # Filter files based on quality score in filename
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

    def get_sampled_files(self, files_list, num_sample):
        if num_sample is not None:
            return random.sample(files_list, num_sample)
        else:
            return files_list

    def __len__(self):
        # Compute total of samples in the dataset
        total_length = sum(len(self.samples[class_name]) for class_name in self.classes)
        return total_length

    def __getitem__(self, idx):
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
        
#         path = os.path.join(self.data_folder, class_folder, file_name)
        path = file_name
        label = class_index
        
        match = re.search(r"-(\d+\.\d+)\.npz", path)
        if match:
            quality_score = float(match.group(1))

        data = numpy.load(path)
        image = data['arr_0']
        
        m, M = numpy.quantile(image, [0.01, 0.995])
        m, M = image.min(), image.max()
        image = (image - m) / (M - m)
        if self.requires_3_channels:
            image = numpy.tile(image[numpy.newaxis], (3, 1, 1))
        else:
            image = image[numpy.newaxis]
        image = torch.tensor(image, dtype=torch.float32)        
        
        if self.transform:
            image = self.transform(image)
        
        return image, {"label" : label, "dataset-idx" : dataset_idx, "score" : quality_score}
    
    def __repr__(self):
        out = "\n"
        for key, values in self.samples.items():
            out += f"{key} - {len(values)}\n"        
        return "Dataset(optim) -- length: {}".format(len(self)) + out
    
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

def get_dataset(name, **kwargs):
    if name == "optim":
        dataset = CreateOptimDataset(
            "./data/ssl/testdata", 
            num_samples={'actin':None, 'tubulin':None, 'CaMKII_Neuron':None, 'PSD95_Neuron':None}, 
            apply_filter=True,
            classes=['actin', 'tubulin', 'CaMKII_Neuron', 'PSD95_Neuron'],
            **kwargs
        )
    elif name == "factin-rings-fibers":
        dataset = CreateFactinRingsFibersDataset(
            "/home-local/Multilabel-Proteins-Actin/Segmentation/precise",
            classes=["rings", "fibers"],
            **kwargs
        )
    elif name == "factin-block-glugly":
        dataset = CreateFActinBlockGluGlyDataset(
            "/home-local/Actin-Dataset/EXP192 (18-09-18) - BlockGluGly",
            classes=["Block", "GLU-GLY"],
            **kwargs
        )
    else:
        raise NotImplementedError(f"`{name}` dataset is not implemented")
    return dataset
