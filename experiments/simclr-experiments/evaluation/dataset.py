
import os
import random
import numpy
import glob
import re 
import torch
import tifffile
import copy

from typing import Any
from torch.utils.data import Dataset
from skimage import filters

LOCAL_CACHE = {}

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
        
        # m, M = image.min(), image.max()
        # image = (image - m) / (M - m)
        m, M = numpy.quantile(image, [0.01, 0.995])
        image = numpy.clip((image - m) / (M - m), 0, 1)
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
            files = glob.glob(os.path.join(data_folder, f"**/{class_name}/*merged.tif"), recursive=True)
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

        # m, M = image.min(), image.max()
        # image = (image - m) / (M - m)
        m, M = numpy.quantile(image, [0.01, 0.995])
        image = numpy.clip((image - m) / (M - m), 0, 1)
        if self.requires_3_channels:
            image = numpy.tile(image[numpy.newaxis], (3, 1, 1))
        else:
            image = image[numpy.newaxis]
        image = torch.tensor(image, dtype=torch.float32)        
        
        if self.transform:
            image = self.transform(image)
        
        return image, {"label" : label, "dataset-idx" : dataset_idx, "path" : path}
    
    def __repr__(self):
        out = "\n"
        for key, values in self.samples.items():
            out += f"{key} - {len(values)}\n"
        return "Dataset(F-actin) -- length: {}".format(len(self)) + out


class CreateFActinDataset(Dataset):

    DATA = {
        "0Mg": {
            "DIV6": [],
            "DIV8": [["EXP175", ""], ["EXP180", "08"], ["EXP186", "05"], ["EXP204", "02"], ["EXP214", "02"]],
            "DIV13": [["EXP180", "09"], ["EXP186", "06"], ["EXP190", ""], ["EXP202", "14"], ["EXP203", "04"], ["EXP210", "07"], ["EXP214", "12"], ["EXP215", "11"], ["EXP217", "17"]],
            "DIV20": []
        },
        "Block": {
            "DIV6": [],
            "DIV8": [["EXP175", ""], ["EXP180", "02"], ["EXP186", "01"], ["EXP192", "01"], ["EXP197", "01"], ["EXP204", "01"], ["EXP214", "01"]],
            "DIV13": [["EXP180", "03"], ["EXP186", "02"], ["EXP190", "01"], ["EXP202", "12"], ["EXP203", "01"], ["EXP210", "04"], ["EXP214", "11"], ["EXP215", "01"], ["EXP217", "01"]],
            "DIV20": []
        },
        "KCl": {
            "DIV6": [],
            "DIV8": [["EXP175", ""], ["EXP180", "05"], ["EXP186", "03"], ["EXP192", "02"], ["EXP197", "02"], ["EXP204", "03"], ["EXP214", "03"]],
            "DIV13": [["EXP180", "06"], ["EXP186", "04"], ["EXP190", "02"], ["EXP202", "02"], ["EXP203", "05"], ["EXP210", "13"], ["EXP214", "13"], ["EXP215", "02"], ["EXP217", "06"]],
            "DIV20": []
        },
        "Glu-Gly": {
            "DIV6": [],
            "DIV8": [["EXP175", ""], ["EXP180", ""], ["EXP186", "07"], ["EXP192", "07"], ["EXP197", "07"], ["EXP204", "04"], ["EXP214", "06"]],
            "DIV13": [["EXP180", ""], ["EXP186", "08"], ["EXP190", ""], ["EXP202", "03"], ["EXP203", "03"], ["EXP210", "11"], ["EXP214", "17"], ["EXP217", "10"]],
            "DIV20": []
        },
    }

    def __init__(self, data_folder : str, transform : Any, classes : list, requires_3_channels : bool=False):
    
        self.div = "DIV13"
        self.data_folder = data_folder
        self.transform = transform
        self.classes = classes
        self.requires_3_channels = requires_3_channels
    
        self.samples = {}
        for class_name in self.classes:
            files = []
            for exp, image_id in self.DATA[class_name][self.div]:
                if image_id:
                    found = sorted(glob.glob(os.path.join(self.data_folder, f"**/*{exp}*/{image_id}*merged.tif"), recursive=True))
                    files.extend(found)
            print(class_name, len(files))
            self.samples[class_name] = self.get_valid_indices(files)
        
    def get_valid_indices(self, files: list, crop_size:int=224):

        def get_dendrite_foreground(img):
            """Gets the foreground of the dendrite channel using a gaussian blur of
            sigma = 20 and the otsu threshold.

            :param img: A 3D numpy

            :returns : A binary 2D numpy array of the foreground
            """
            blurred = filters.gaussian(img[2], sigma=20)
            blurred /= blurred.max()
            val = filters.threshold_otsu(blurred)
            return (blurred > val).astype(int)

        def get_label_ratio(predRings, predFilaments, dendrite, r_thres=0.2, f_thres=0.4):
            """
            Computes the ratio of the labeling in the dendrite

            :param predRings: The prediction of the rings from the network
            :param predFilaments: The prediction of the filaments from the network
            :param dendrite: The dendrite mask
            :param r_thres: Default threshold for the rings (obtained from the ROC curve)
            :param f_thres: Default threshold for the filaments (obtained from the ROC curve)

            :returns : A list of ratios
            """
            predRatioRings = ((predRings > r_thres) * dendrite).sum() / dendrite.sum()
            predRatioFilaments = ((predFilaments > f_thres) * dendrite).sum() / dendrite.sum()
            return [predRatioRings, predRatioFilaments]

        out = []
        for file in files:
            image = tifffile.imread(file)

            # Calculates ratios from images
            if file in LOCAL_CACHE:
                ratios = LOCAL_CACHE[file]
            else:
                ring_file = file.replace("merged.tif", "merged_regression124_predRings.tif")
                ring = tifffile.imread(ring_file)
                filament_file = file.replace("merged.tif", "merged_regression124_predFilaments.tif")
                filament = tifffile.imread(filament_file)
                dendrite = get_dendrite_foreground(image)
                ratios = get_label_ratio(ring, filament, dendrite)
                LOCAL_CACHE[file] = ratios

            m, M = numpy.quantile(image[0], [0.01, 0.995])
            # Dendrite foreground
            threshold = filters.threshold_otsu(image[2])
            foreground = image[2] > threshold
            for j in range(0, image.shape[-2], int(0.75 * crop_size)):
                for i in range(0, image.shape[-1], int(0.75 * crop_size)):
                    slc = (
                        slice(j, j + crop_size) if j + crop_size < image.shape[-2] else slice(image.shape[-2] - crop_size, image.shape[-2]),
                        slice(i,  i + crop_size) if i + crop_size < image.shape[-1] else slice(image.shape[-1] - crop_size, image.shape[-1]),
                    )                    
                    crop = foreground[slc]
                    if crop.sum() > 0.1 * crop.size:
                        out.append({
                            "path" : file,
                            "slc" : slc,
                            "minmax" : (m, M),
                            "ratios" : ratios
                        })
        return out
    
    def __len__(self):
        return sum(len(value) for value in self.samples.values())
    
    def __getitem__(self, idx : int):
        
        class_name = None
        dataset_idx = idx

        for i, class_name in enumerate(self.classes):
            if idx < len(self.samples[class_name]):
                info = self.samples[class_name][idx]
                label = i
                break
            else:
                idx -= len(self.samples[class_name])

        path = info["path"]
        slc = info["slc"]
        m, M = info["minmax"]
        
        image = tifffile.imread(path)[0]
        # m, M = image.min(), image.max()
        # m, M = numpy.quantile(image, [0.01, 0.995])

        image = image[slc]
        image = numpy.clip((image - m) / (M - m), 0, 1)

        if self.requires_3_channels:
            image = numpy.tile(image[numpy.newaxis], (3, 1, 1))
        else:
            image = image[numpy.newaxis]
        image = torch.tensor(image, dtype=torch.float32)        
        
        if self.transform:
            image = self.transform(image)
        
        info = copy.deepcopy(info)
        info.pop("slc")
        return image, {"label" : label, "dataset-idx" : dataset_idx, **info}
    
    def __repr__(self):
        out = "\n"
        for key, values in self.samples.items():
            out += f"{key} - {len(values)}\n"
        return "Dataset(F-actin) -- length: {}".format(len(self)) + out

class CreateMitoDataset(Dataset):
    def __init__(self, data_folder : str, transform : Any, classes : list, requires_3_channels : bool=False):
        super().__init__()

        self.data_folder = data_folder
        self.transform = transform
        self.classes = classes
        self.requires_3_channels = requires_3_channels

        to_remove = [
            "L25_AIW_msTH488_RbTom20_635_Insert_04_to_segment.tif",
            "L26_3x_msTH488_RbTom20_635_Insert_02_.tif",
            "L30_A53T_msTH488_RbTom20_635_Insert_02_.tif",
            "L31_2KO_msTH488_RbTom20_635_Insert_02_.tif",
            "L32_A53T_msTH488_RbTom20_635_Insert_cs1_01_.tif",
            "L33_A53T_msTH488_RbTom20_635_Insert_cs1_04_.tif",
        ]
    
        self.samples = {}
        for class_name in self.classes:
            files = []
            found = sorted(glob.glob(os.path.join(self.data_folder, f"**/{class_name}/*.tif"), recursive=True))
            files.extend(found)
            files = [file for file in files if os.path.basename(file) not in to_remove]
            
            print(class_name, len(files))
            self.samples[class_name] = self.get_valid_indices(files)
        
    def get_valid_indices(self, files: list, crop_size:int=224):

        out = []
        for file in files:
            image = tifffile.imread(file)

            m, M = numpy.quantile(image[1], [0.01, 0.995])
            # Dendrite foreground
            threshold = filters.threshold_otsu(image[0])
            foreground = image[0] > threshold
            for j in range(0, image.shape[-2], crop_size):
                for i in range(0, image.shape[-1], crop_size):
                    slc = (
                        slice(j, j + crop_size) if j + crop_size < image.shape[-2] else slice(image.shape[-2] - crop_size, image.shape[-2]),
                        slice(i,  i + crop_size) if i + crop_size < image.shape[-1] else slice(image.shape[-1] - crop_size, image.shape[-1]),
                    )
                    crop = foreground[slc]
                    if crop.sum() > 0.01 * crop.size:
                        out.append({
                            "path" : file,
                            "slc" : slc,
                            "minmax" : (m, M)
                        })
        return out
    
    def __len__(self):
        return sum(len(value) for value in self.samples.values())
    
    def __getitem__(self, idx : int):
        
        class_name = None
        dataset_idx = idx

        for i, class_name in enumerate(self.classes):
            if idx < len(self.samples[class_name]):
                info = self.samples[class_name][idx]
                label = i
                break
            else:
                idx -= len(self.samples[class_name])

        path = info["path"]
        slc = info["slc"]
        m, M = info["minmax"]
        
        image = tifffile.imread(path)[1]
        # m, M = image.min(), image.max()
        # m, M = numpy.quantile(image, [0.01, 0.995])

        image = image[slc]
        image = numpy.clip((image - m) / (M - m), 0, 1)

        if self.requires_3_channels:
            image = numpy.tile(image[numpy.newaxis], (3, 1, 1))
        else:
            image = image[numpy.newaxis]
        image = torch.tensor(image, dtype=torch.float32)        
        
        if self.transform:
            image = self.transform(image)
        
        info = copy.deepcopy(info)
        info.pop("slc")
        return image, {"label" : label, "dataset-idx" : dataset_idx, **info}
    
    def __repr__(self):
        out = "\n"
        for key, values in self.samples.items():
            out += f"{key} - {len(values)}\n"
        return "Dataset(Mitochondria) -- length: {}".format(len(self)) + out

def get_dataset(name, **kwargs):
    if name == "optim":
        dataset = CreateOptimDataset(
            "./data/SSL/testdata", 
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
    elif name == "factin-block-kcl-glugly":
        dataset = CreateFActinBlockGluGlyDataset(
            "/home-local/Actin-Dataset/EXP192 (18-09-18)",
            classes=["Block", "KCl", "GLU-GLY"],
            **kwargs
        )        
    elif name == "factin":
        dataset = CreateFActinDataset(
            "/home-local/Actin-Dataset/Dataset_Pour_Anthony_13-09-18",
            classes=["Block", "0Mg", "KCl", "Glu-Gly"],
            **kwargs
        )
    elif name == "mito":
        dataset = CreateMitoDataset(
            "/home-local2/projects/FLCDataset/oferguson/Inserts_Images_and_Masks",
            classes = ["3x", "2KO", "A53T", "AIW"],
            **kwargs
        )        
    elif name == "mito-3x-2ko":
        dataset = CreateMitoDataset(
            "/home-local2/projects/FLCDataset/oferguson/Inserts_Images_and_Masks",
            classes = ["3x", "2KO"],
            **kwargs
        )
    elif name == "mito-a53t-aiw":
        dataset = CreateMitoDataset(
            "/home-local2/projects/FLCDataset/oferguson/Inserts_Images_and_Masks",
            classes = ["A53T", "AIW"],
            **kwargs
        )        
    else:
        raise NotImplementedError(f"`{name}` dataset is not implemented")
    return dataset
