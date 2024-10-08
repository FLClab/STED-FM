import tarfile 
import numpy
import numpy as np
import io
import torch
import skimage.transform
from typing import Any, List, Tuple, Callable
from torch.utils.data import Dataset, get_worker_info
from tqdm import tqdm
from torchvision import transforms
import h5py
import random
import os 
import glob
import re
import tifffile
from collections import defaultdict
import copy
from skimage import filters
from PIL import Image
from zipfile import ZipFile

from DEFAULTS import BASE_PATH
from dataset_builder import condition_dict

LOCAL_CACHE = {}

def get_dataset(name: str, path: str, **kwargs):
    if name == "CTC":
        dataset = CTCDataset(path, **kwargs)
    elif name == "JUMP":
        dataset = TarJUMPDataset(path, **kwargs)
    elif name == "STED": 
        dataset = TarFLCDataset(path, **kwargs)
    elif name == "optim":
        dataset = OptimDataset(
            os.path.join(BASE_PATH, "evaluation-data", "optim-data"), 
            num_samples={'actin':None, 'tubulin':None, 'CaMKII_Neuron':None, 'PSD95_Neuron':None}, 
            apply_filter=True,
            classes=['actin', 'tubulin', 'CaMKII_Neuron', 'PSD95_Neuron'],
            **kwargs
        )
    elif name == "HPA":
        dataset = HPADataset(path, **kwargs)
    elif name == "factin":
        dataset = CreateFActinDataset(
            os.path.join(BASE_PATH, "evaluation-data", "actin-data"),
            classes=["Block", "0Mg", "KCl", "Glu-Gly"],
            **kwargs
        )        
    elif name == "mito-3x-2ko":
        dataset = CreateMitoDataset(
            os.path.join(BASE_PATH, "evaluation-data", "mito-data"),
            classes = ["3x", "2KO"],
            **kwargs
        )
    elif name == "mito-a53t-aiw":
        dataset = CreateMitoDataset(
            os.path.join(BASE_PATH, "evaluation-data", "mito-data"),
            classes = ["A53T", "AIW"],
            **kwargs
        )        
    else:
        raise NotImplementedError(f"Dataset `{name}` not implemented yet.")
    return dataset

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

    def __init__(self, data_folder : str, transform : Any, classes : list, n_channels : int=1):
    
        self.div = "DIV13"
        self.data_folder = data_folder
        self.transform = transform
        self.classes = classes
        self.n_channels = n_channels
    
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

        image = numpy.tile(image[numpy.newaxis], (self.n_channels, 1, 1))
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
    def __init__(self, data_folder : str, transform : Any, classes : list, n_channels : bool=False):
        super().__init__()

        self.data_folder = data_folder
        self.transform = transform
        self.classes = classes
        self.n_channels = n_channels

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

        image = numpy.tile(image[numpy.newaxis], (self.n_channels, 1, 1))
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

class CreateFactinRingsFibersDataset(Dataset):
    def __init__(self, data_folder : str, transform : Any, classes : list, n_channels : bool=False):
    
        self.data_folder = data_folder
        self.transform = transform
        self.classes = classes
        self.n_channels = n_channels
        
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
        image = np.tile(image[np.newaxis], (self.n_channels, 1, 1))
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
    def __init__(self, data_folder : str, transform : Any, classes : list, n_channels : bool=False):
    
        self.data_folder = data_folder
        self.transform = transform
        self.classes = classes
        self.n_channels = n_channels
        
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
        image = np.tile(image[np.newaxis], (self.n_channels, 1, 1))
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
            min_quality_score: float = 0.70,
            *args, **kwargs
    ) -> None:
        self.data_folder = data_folder
        self.num_samples = num_samples
        self.transform = transform
        self.apply_filter = apply_filter
        self.classes = classes 
        self.num_classes = len(self.classes)
        self.n_channels = n_channels
        self.class_files = {}
        self.samples = {}
        self.num_classes = len(classes)
        self.min_quality_score = min_quality_score

        self.labels = []
        for i, class_name in enumerate(classes):
            class_folder = os.path.join(data_folder, class_name)
            self.class_files[class_name] = self._filter_files(class_folder)
            self.samples[class_name] = self._get_sampled_files(self.class_files[class_name], self.num_samples.get(class_name))
            self.labels.extend([i] * len(self.samples[class_name]))

    def _filter_files(self, class_folder):
        files = glob.glob(os.path.join(class_folder, "**/*.npz"), recursive=True)
        filtered_files = []
        for file in files:
            match = re.search(r"-(\d+\.\d+)\.npz", file)
            if match:
                quality_score = float(match.group(1))
                if not self.apply_filter or quality_score >= self.min_quality_score:
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
        # image = np.tile(image[np.newaxis], (self.n_channels, 1, 1))
        # image = torch.tensor(image, dtype=torch.float32)   
        if self.n_channels == 3:
                img = np.tile(image[np.newaxis], (3, 1, 1))
                img = np.moveaxis(img, 0, -1)
                img = transforms.ToTensor()(img)
                img = transforms.Normalize(mean=[0.0695771782959453, 0.0695771782959453, 0.0695771782959453], std=[0.12546228631005282, 0.12546228631005282, 0.12546228631005282])(img)
        else:
            img = transforms.ToTensor()(image)
        
        if self.transform:
            img = self.transform(img)
        
        # label = np.float64(label)
        return img, {"label" : label, "dataset-idx" : dataset_idx, "score" : quality_score}

    def __repr__(self):
        out = "\n"
        for key, values in self.samples.items():
            out += f"{key} - {len(values)}\n"        
        return "Dataset(optim) -- length: {}".format(len(self)) + out
    
class PeroxisomeDataset(Dataset):
    """
    Note. We do not use "6hbackGluc" since it is not present as a Triplo in the dataset
    """
    def __init__(
        self, source:str, 
        transform: Any, 
        classes: List = ["6hGluc", "4hMeOH", "6hMeOH", "8hMeOH", "16hMeOH"], 
        n_channels: int = 1,
        resize_mode : str = "pad",
        superclasses: bool = False,
        num_samples: int = None,
        **kwargs
    ): 
        super().__init__()
        self.source = source
        self.transform = transform
        self.classes = classes
        self.n_channels = n_channels
        self.resize_mode = resize_mode
        self.num_classes = len(self.classes)

        self.samples = {}
        with open(source, "r") as file:
            files = file.readlines()
            files = [os.path.join(BASE_PATH, file.strip()[1:]) for file in files]
        for i, class_name in enumerate(self.classes):
            self.samples[class_name] = self._get_sampled_files(files_list=[f for f in files if class_name in f], num_sample=num_samples)

        if superclasses:
            self.__merge_superclasses()
        self.info = self.__get_info()

        print("----------")
        for k in self.samples.keys():
            print(f"Class {k} samples: {len(self.samples[k])}")
        print("----------")

    def _get_sampled_files(self, files_list, num_sample):
        if num_sample is not None:
            return random.sample(files_list, num_sample)
        else:
            return files_list

    def __merge_superclasses(self) -> None:
        merged_samples = defaultdict(list)
        for key in self.samples.keys():
            if "gluc" in key.lower():
                merged_samples["gluc"].extend(self.samples[key])
            elif "meoh" in key.lower():
                merged_samples["meoh"].extend(self.samples[key])
            else:
                continue
        self.samples = merged_samples
        self.classes = ["gluc", "meoh"]
        self.num_classes = len(self.classes)

    def __get_info(self):
        info = []
        for key, values in self.samples.items():
            for value in values:
                info.append({
                    "img" : value,
                    "label" : key
                })
        return info
    
    def __getitem__(self, idx: int):
        item = self.info[idx]

        img = tifffile.imread(item["img"])[0] # We only select Pex3 channel
        m, M = img.min(), img.max()
        img = (img - m) / (M - m)

        # Images do not match the expected 224x224 size
        if self.resize_mode == "pad":
            img = numpy.pad(img, ((0, 224 - img.shape[0]), (0, 224 - img.shape[1])), mode="constant", constant_values=0)
        elif self.resize_mode == "resize":
            img = skimage.transform.resize(img, (224, 224), order=1, mode="constant", cval=0, anti_aliasing=True, preserve_range=True)

        label = self.classes.index(item["label"])

        if self.n_channels == 3:
            img = np.tile(img[np.newaxis, :], (3, 1, 1))
            img = torch.tensor(img, dtype=torch.float32)
            # img = transforms.Normalize(mean=[0.0695771782959453, 0.0695771782959453, 0.0695771782959453], std=[0.12546228631005282, 0.12546228631005282, 0.12546228631005282])(img)
            img = transforms.Normalize(mean=[0.07, 0.07, 0.07], std=[0.03, 0.03, 0.03])(img)
        else:
            img = torch.tensor(img[np.newaxis, :], dtype=torch.float32)
            img = self.transform(img) if self.transform is not None else img         
        return img, {"label" : label, "dataset-idx" : idx}

    def __len__(self):
        return len(self.info)
    
class PolymerRingsDataset(Dataset):
    """
    Dataset containing ESCRT-III polymer rings (CdvB, CdvB1 and CdvB2) in wildtype archaea (Sacidocaldarius_DSM639) 
    imaged with STimulated Emission Depletion (STED) nanoscopy.

    The task is to classify CdvB1 and CdvB2 rings when CdvB is present or not.

    When superclass is activated the task consists only in classifying between CdvB1 and CdvB2 rings.
    """
    def __init__(
        self, source:str, 
        transform: Any, 
        classes: List = ["CdvB1", "CdvB2"], 
        n_channels: int = 1,
        resize_mode : str = "pad",
        superclasses: bool = True,
        num_samples: int = None,
        **kwargs
    ): 
        super().__init__()
        self.source = source
        self.transform = transform
        self.classes = classes
        self.n_channels = n_channels
        self.resize_mode = resize_mode
        self.num_classes = len(self.classes)

        with open(source, "r") as file:
            files = file.readlines()
            files = [os.path.join(BASE_PATH, file.strip()[1:]) for file in files]
        
        self.samples = {}        
        for i, class_name in enumerate(self.classes):
            self.samples[class_name] = self._get_sampled_files(files_list=[f for f in files if class_name in f], num_sample=num_samples)
        
        if not superclasses:
            tmp = {}
            for name in ["with_CdvB", "no_CdvB"]:
                for key, values in self.samples.items():
                    tmp[f"{key} ({name})"] = [file for file in values if name in file]
            self.samples = tmp 

        print("Samples: ", os.path.basename(source))
        for key, values in self.samples.items():
            print(key, len(values))

        self.classes = list(sorted(self.samples.keys()))
        self.num_classes = len(self.classes)
        self.info = self.__get_info()

        print("----------")
        for k in self.samples.keys():
            print(f"Class {k} samples: {len(self.samples[k])}")
        print("----------")

        # statistics = defaultdict(list)
        # for i in range(len(self.info)):
        #     item = self.info[i]
        #     img = tifffile.imread(item["img"])[item["chan-idx"]]
        #     m, M = img.min(), img.max()
        #     img = (img - m) / (M - m)
        #     statistics["mean"].append(numpy.mean(img))
        #     statistics["std"].append(numpy.std(img))
        # print(f"Mean: {numpy.mean(statistics['mean'])}, Std: {numpy.mean(statistics['std'])}")

    def _get_sampled_files(self, files_list, num_sample):
        if num_sample is not None:
            return random.sample(files_list, num_sample)
        else:
            return files_list

    def __get_info(self):
        info = []
        for key, values in self.samples.items():
            protein_id = key.split(" ")[0] if "(" in key else key

            for value in values:
                basename = os.path.basename(value)
                name = os.path.splitext(basename)[0]

                chan = name.split("_")
                if protein_id in chan:
                    chan_idx = chan.index(protein_id) - 3
                else:
                    continue
                
                info.append({
                    "img" : value,
                    "label" : key,
                    "chan-idx" : chan_idx
                })
        return info
    
    def __getitem__(self, idx: int):
        item = self.info[idx]

        img = tifffile.imread(item["img"])[item["chan-idx"]]
        m, M = img.min(), img.max()
        img = (img - m) / (M - m)

        # Images do not match the expected 224x224 size
        if self.resize_mode == "pad":
            img = numpy.pad(img, ((0, 224 - img.shape[0]), (0, 224 - img.shape[1])), mode="constant", constant_values=0)
        elif self.resize_mode == "resize":
            img = skimage.transform.resize(img, (224, 224), order=1, mode="constant", cval=0, anti_aliasing=True, preserve_range=True)

        label = self.classes.index(item["label"])

        if self.n_channels == 3:
            img = np.tile(img[np.newaxis, :], (3, 1, 1))
            img = torch.tensor(img, dtype=torch.float32)
            # img = transforms.Normalize(mean=[0.0695771782959453, 0.0695771782959453, 0.0695771782959453], std=[0.12546228631005282, 0.12546228631005282, 0.12546228631005282])(img)
            img = transforms.Normalize(mean=[0.03, 0.03, 0.03], std=[0.09, 0.09, 0.09])(img)
        else:
            img = torch.tensor(img[np.newaxis, :], dtype=torch.float32)
            img = self.transform(img) if self.transform is not None else img      
        return img, {"label" : label, "dataset-idx" : idx}

    def __len__(self):
        return len(self.info)    

class NeuralActivityStates(Dataset):
    def __init__(
            self,
            h5file: str,
            transform: Callable = None,
            n_channels: int = 1,
            num_samples: int = None,
            num_classes: int = 4,
            protein_id: int = 3,
            balance: bool = True,
    ) -> None:
        self.h5file = h5file 
        self.transform = transform 
        self.n_channels = n_channels
        self.num_samples = num_samples 
        self.num_classes = num_classes 

        with h5py.File(h5file, "r") as handle:
            images = handle["images"][()]
            conditions = handle["conditions"][()]
            proteins = handle["proteins"][()]

        protein_mask = np.where(proteins == protein_id)
        self.images = images[protein_mask]
        self.labels = conditions[protein_mask]
        self.proteins = proteins[protein_mask]

        # print(f"{numpy.mean(numpy.mean(self.images, axis=(1, 2)))=}")
        # print(f"{numpy.mean(numpy.std(self.images, axis=(1, 2)))=}")

        KEEPCLASSES = [0, 1, 2, 3]

        self.num_classes = len(KEEPCLASSES)
        mask = np.isin(self.labels, KEEPCLASSES)
        self.images = self.images[mask]
        self.labels = self.labels[mask]
        self.proteins = self.proteins[mask]
        self.classes = []
        for i in KEEPCLASSES:
            for key, value in condition_dict.items():
                if i == value:
                    self.classes.append(key)
                    break

        self.__reset_labels() #  Only required if we're not using KEEPCLASSES = [0, 1, 2]


        assert self.images.shape[0] == self.labels.shape[0] == self.proteins.shape[0]
        
        if balance:
            self.__balance_classes()
        self.dataset_size = self.images.shape[0]

    def __reset_labels(self) -> None:
        unique = np.unique(self.labels)
        new_labels = np.zeros_like(self.labels)
        for i, u in enumerate(unique):
            mask = self.labels == u 
            new_labels[mask] = i
        self.labels = new_labels        

    def __balance_classes(self) -> None:
        uniques, counts = np.unique(self.labels, return_counts=True) 
        minority_count, minority_class = np.min(counts), np.argmin(counts)
        indices = []
        if self.num_samples is not None:
            minority_count = self.num_samples

        for unique in uniques:
            ids = np.where(self.labels == unique)[0]
            ids = np.random.choice(ids, size=minority_count)
            indices.extend(ids)
        indices = np.sort(indices)
        print("\n************************************")
        print(f"Sampling ids = {indices}")
        print("************************************\n")
        self.images = self.images[indices]
        self.labels = self.labels[indices]
        self.proteins = self.proteins[indices]

    def __len__(self) -> int:
        return self.dataset_size 
    

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        img, protein, label = self.images[idx], self.proteins[idx], self.labels[idx]
        if self.n_channels == 3:
            img = np.tile(img[np.newaxis, :], (3, 1, 1))
            img = torch.tensor(img, dtype=torch.float32)
            # img = transforms.Normalize(mean=[0.0695771782959453, 0.0695771782959453, 0.0695771782959453], std=[0.12546228631005282, 0.12546228631005282, 0.12546228631005282])(img)
            img = transforms.Normalize(mean=[0.014, 0.014, 0.014], std=[0.03, 0.03, 0.03])(img)

        else:
            img = torch.tensor(img[np.newaxis, :], dtype=torch.float32)
            img = self.transform(img) if self.transform is not None else img 
        return img, {"label": label, "protein": protein}

class ProteinDataset(Dataset):
    def __init__(
            self, 
            h5file: str, 
            class_ids: List[int] = None, 
            class_type: str = "proteins", 
            transform = None,
            n_channels: int = 1,
            num_samples: int = None,
            num_classes : int = 4
            ) -> None:
        self.h5file = h5file 
        self.class_ids = class_ids
        self.class_type = class_type
        self.n_channels = n_channels
        self.num_samples = num_samples
        self.num_classes = num_classes

        if self.num_samples is None:
            with h5py.File(h5file, "r") as hf:
                self.dataset_size = int(hf[self.class_type].size)
                self.labels = hf[self.class_type][()]
        else:
            with h5py.File(h5file, "r") as hf:
                indices = []
                labels = hf[self.class_type][()]
                for i in range(num_classes):
                    inds = np.argwhere(np.array(labels) == i)
                    inds = np.random.choice(inds.ravel(), size=num_samples, replace=True)
                    indices.append(inds)
                    label_ids = np.sort(np.concatenate([ids.ravel() for ids in indices]).astype('int'))
                    self.labels = hf["proteins"][label_ids]
                    self.images = hf["images"][label_ids]
                    self.conditions = hf["conditions"][label_ids]
                    self.dataset_size = self.labels.shape[0]
            
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        with h5py.File(self.h5file, "r") as hf:
            if self.num_samples == None:
                img = hf["images"][idx]
                protein = hf["proteins"][idx]
                condition = hf["conditions"][idx]
            else:
                img = self.images[idx]
                protein = self.labels[idx]
                condition = self.conditions[idx]

            if self.n_channels == 3:
                img = np.tile(img[np.newaxis], (3, 1, 1))
                img = np.moveaxis(img, 0, -1)
                img = transforms.ToTensor()(img)
                img = transforms.Normalize(mean=[0.0695771782959453, 0.0695771782959453, 0.0695771782959453], std=[0.12546228631005282, 0.12546228631005282, 0.12546228631005282])(img)
            else:
                img = transforms.ToTensor()(img)
        l = protein if self.class_type == "proteins" else condition
        other = condition if self.class_type == "proteins" else protein
        return img, {"label": l, "condition": other}

class CTCDataset(Dataset):
    """
    Dataset class for loading and processing image data from a HDF5 file.
    This dataset is specifically designed for the CTC dataset.
    """
    def __init__(
            self,
            h5file: str,
            n_channels: int = 1,
            transform: Any = None,
            return_metadata: bool = False,
            **kwargs
    ) -> None:
        """
        Instantiates a new ``CTCDataset`` object.

        :param h5file: The path to the HDF5 file to load data from.
        :param n_channels: The number of channels in the image data.
        :param transform: The transformation to apply to the image data.
        """
        self.h5file = h5file
        self.n_channels = n_channels
        self.transform = transform
        self.return_metadata = return_metadata
        with h5py.File(h5file, "r") as hf:
            self.dataset_size = int(hf["protein"].size)

    def __len__(self):
        """
        Implements the ``__len__`` method for the dataset.
        """
        return self.dataset_size

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Implements the ``__getitem__`` method for the dataset.

        :param idx: The index of the item to retrieve.

        :returns : The item at the given index.
        """
        with h5py.File(self.h5file, "r") as hf:
            img = hf["images"][idx]
            protein = hf['proteins'][idx]
            condition = hf['condition'][idx]

            img = img[np.newaxis]
            img = torch.tensor(img, dtype=torch.float32)
            if self.transform is not None:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
        if self.return_metadata:
            return img, {"protein": protein, "condition": condition}
        return img

class JUMPCPDataset(Dataset):
    def __init__(
            self, 
            h5file: str, 
            n_channels: int = 1, 
            transform: Callable = None, 
            use_cache: bool = False,
            max_cache_size: int = 128e9,
            cache_system: str = None,
            return_metadata: bool = None,
            world_size: int =1, 
            rank: int = 0,
            **kwargs
            ):
        self.h5file = h5file 
        self.n_channels = n_channels
        self.transform = transform
        self.__cache = {}
        self.__max_cache_size = max_cache_size 
        self.return_metadata = return_metadata
        self.world_size = world_size
        self.rank = rank
        self.dataset_size = 1300008

        worker = get_worker_info()
        worker = worker.id if worker else None 
        
        indices = np.arange(0, self.dataset_size, 1)

        self.members = self.__setup_multiprocessing(indices)
        if use_cache and self.__max_cache_size >0:
            self.__cache_size = 0
            if cache_system is not None:
                self.__cache = cache_system
        self.__fill_cache()

    def __getsizeof(self, obj: Any) -> int:
        """
        Implements a simple function to estimate the size of an object in memory.

        :param obj: The object to estimate the size of.

        :returns : The size of the object in bytes.
        """
        if isinstance(obj, dict):
            return sum([self.__getsizeof(o) for o in obj.values()])
        elif isinstance(obj, (list, tuple)):
            return sum([self.__getsizeof(o) for o in obj])
        elif isinstance(obj, str):
            return len(str)
        else:
            return obj.size * obj.dtype.itemsize


    def __setup_multiprocessing(self, members : np.ndarray):
        """
        Setup multiprocessing for the dataset.

        :param members: The list of members to setup multiprocessing for.

        :returns : A `list` of members.
        """
        if self.world_size > 1:
            num_members = len(members)
            num_members_per_gpu = num_members // self.world_size
            members = members[self.rank * num_members_per_gpu : (self.rank + 1) * num_members_per_gpu]
        return members
    
    def __fill_cache(self):
        """
        Implements a function to fill up the cache with data from the TarFile.
        """
        indices = np.arange(0, len(self.members), 1)
        np.random.shuffle(indices)
        print("Filling up the cache...")
        pbar = tqdm(indices, total=indices.shape[0])
        with h5py.File(self.h5file, "r") as hf:
            for idx in pbar:
                if self.__cache_size >= self.__max_cache_size:
                    break
                data = hf["images"][idx]
                self.__cache[idx] = data
                self.__cache_size += self.__getsizeof(data)
                pbar.set_description(f"Cache size --> {self.__cache_size * 1e-9:0.2f}G")

    def __len__(self):
        return len(self.members)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx in self.__cache:
            img = self.__cache[idx]
        else:
            with h5py.File(self.h5file, "r") as hf:
                img = hf['images'][idx]
        if self.transform is not None:
            img = self.transform(img).float()
        else:
            img = img[np.newaxis]
            img = torch.tensor(img, dtype=torch.float32)
        return img
    
class ArchiveDataset(Dataset):
    """
    This is an attempt at reading both zip and tar from a single class.

    Some methods should be reimplemented in daughter classes.
    """

    READERS = {
        ".zip" : ZipFile,
        ".tar" : tarfile.open
    }

    def __init__(self, 
                 archive_path, 
                 use_cache=False, 
                 max_cache_size=16e9, 
                 transform: Any=None, 
                 cache_system=None, 
                 world_size=1, 
                 rank=0, 
                 **kwargs
        ):
        
        super(ArchiveDataset, self).__init__()

        self.__cache = {}
        self.__max_cache_size = max_cache_size
        self.archive_path = archive_path
        self.transform = transform

        # Multiprocessing settings for multi-gpu training
        self.world_size = world_size
        self.rank = rank

        # Archive reader
        ext = os.path.splitext(self.archive_path)[1]
        if ext not in self.READERS:
            raise NotImplementedError(f"Archive type `{ext}` is not implemented. The only supported archives are: {self.READERS.keys()}")
        self.archive_reader = self.READERS[ext]

        # Instantiates the archive object
        worker = get_worker_info()
        worker = worker.id if worker else None
        self.archive_obj = {worker: self.archive_reader(self.archive_path, "r")}    
        
        members = self.get_members()
        self.members = self.__setup_multiprocessing(members)

        if use_cache and self.__max_cache_size > 0:
            self.__cache_size = 0
            if not cache_system is None:
                self.__cache = cache_system
            self.__fill_cache()

    def get_members(self):
        raise NotImplementedError("Implement in daughter class")   

    def get_item_from_archive(self, member):
        raise NotImplementedError("Implement in daughter class")
    
    def get_reader(self):
        # ensure a unique file handle per worker, in multiprocessing settings
        worker = get_worker_info()
        worker = worker.id if worker else None
        if worker not in self.archive_obj:
            self.archive_obj[worker] = self.archive_reader(self.archive_path, "r")
        return self.archive_obj[worker]
    
    def get_data(self, idx):
        if idx in self.__cache:
            data = self.__cache[idx]
        else:
            data = self.get_item_from_archive(self.members[idx])
        return data

    def __getsizeof(self, obj: Any) -> int:
        """
        Implements a simple function to estimate the size of an object in memory.

        :param obj: The object to estimate the size of.

        :returns : The size of the object in bytes.
        """
        if isinstance(obj, dict):
            return sum([self.__getsizeof(o) for o in obj.values()])
        elif isinstance(obj, (list, tuple)):
            return sum([self.__getsizeof(o) for o in obj])
        elif isinstance(obj, str):
            return len(str)
        else:
            return obj.size * obj.dtype.itemsize
    
    def __fill_cache(self):
        """
        Implements a function to fill up the cache with data from the TarFile.
        """
        indices = np.arange(0, len(self.members), 1)
        np.random.shuffle(indices)
        print("Filling up the cache...")
        # pbar = tqdm(indices, total=indices.shape[0])
        for n, idx in enumerate(indices):
            if self.__cache_size >= self.__max_cache_size:
                break
            data = self.get_item_from_archive(self.members[idx])
            self.__cache[idx] = data
            self.__cache_size += self.__getsizeof(data)
            # pbar.set_description(f"Cache size --> {self.__cache_size * 1e-9:0.2f}G")
            if n % 1000 == 0:
                worker = get_worker_info()
                worker = worker.id if worker else None
                print(f"Current cache (worker: {worker} | rank: {self.rank}): {n}/{len(indices)} ({self.__cache_size * 1e-9:0.2f}G/{self.__max_cache_size * 1e-9:0.2f}G)")   

    def __setup_multiprocessing(self, members : list):
        """
        Setup multiprocessing for the dataset.

        :param members: The list of members to setup multiprocessing for.

        :returns : A `list` of members.
        """
        if self.world_size > 1:
            num_members = len(members)
            num_members_per_gpu = num_members // self.world_size

            # members = members[self.rank * num_members_per_gpu : (self.rank + 1) * num_members_per_gpu]
            # Since the members are sorted, it makes more sense to take every `world_size` items
            members = members[self.rank:num_members_per_gpu*self.world_size:self.world_size]
        return members

    def __len__(self):
        """
        Implements the ``__len__`` method for the dataset.
        """
        return len(self.members)
    
    def __getitem__(self, idx):
        return self.get_data(idx)
    
    def __del__(self):
        """
        Close the ZipFile file handles on exit.
        """
        for o in self.archive_obj.values():
            o.close()
            
    def __getstate__(self) -> dict:
        """
        Serialize without the ZipFile references, for multiprocessing compatibility.
        """
        state = dict(self.__dict__)
        state['archive_obj'] = {}
        return state  

class TarJUMPDataset(ArchiveDataset):
    def __init__(
            self,
            tar_path: str,
            use_cache: bool = False,
            max_cache_size: int = 16e9,
            image_channels: int = 1,
            transform: Callable = None,
            cache_system=None,
            world_size: int = 1,
            rank: int = 0,
            *args, **kwargs
    ) -> None:
        super(TarJUMPDataset, self).__init__(
            tar_path,
            use_cache=use_cache,
            max_cache_size=max_cache_size,
            transform=transform,
            cache_system=cache_system,
            world_size=world_size,
            rank=rank
        )
        self.image_channels = image_channels

    def get_members(self):
        return list(sorted(self.get_reader().getmembers(), key=lambda m: m.name))
    
    def get_item_from_archive(self, member: tarfile.TarInfo):
        buffer = io.BytesIO()
        buffer.write(self.get_reader().extractfile(member).read())
        buffer.seek(0)
        data = np.load(buffer, allow_pickle=True)
        data = {key: values for key, values in data.items()}
        return data 

    def __getitem__(self, idx: int) -> torch.Tensor:
        data = self.get_data(idx)
        img = data["image"]
        img = img / 255. 
        if self.transform is not None:
            # This assumes that there is a conversion to torch Tensor in the given transform
            img = self.transform(img)
            if isinstance(img, list):
                img = [x.float() for x in img]
            else:
                img = img.float()
        else:
            if img.ndim == 2:
                img = img[np.newaxis]
            img = torch.tensor(img, dtype=torch.float32)

        return img
            

class TarFLCDataset(ArchiveDataset):
    """
    Dataset class for loading and processing image data from a TarFile object.
    """
    def __init__(self, tar_path: str, 
                 use_cache: bool = False, 
                 max_cache_size: int = 16e9, 
                 image_channels: int = 1, 
                 transform: Any = None, 
                 cache_system=None, 
                 return_metadata: bool=False,
                 world_size: int = 1,
                 rank: int = 0,
                 debug: bool = False,
                 *args, **kwargs
                 ) -> None:
        """
        Instantiates a new ``TarFLCDataset`` object.

        :param tar_path: The path to the TarFile object to load data from.
        :param use_cache: Whether to use a cache system to store data.
        :param max_cache_size: The maximum size of the cache in bytes.
        :param image_channels: The number of channels in the image data.
        :param transform: The transformation to apply to the image data.
        :param cache_system: The cache system to use for storing data. This is 
            used to share a cache system across multiple workers using ``multiprocessing.Manager``.
        :param return_metadata: Whether to return metadata along with the image data.
        """
        self.image_channels = image_channels
        self.return_metadata = return_metadata
        self.debug = debug        

        super(TarFLCDataset, self).__init__(
            tar_path, 
            use_cache=use_cache, 
            max_cache_size=max_cache_size, 
            transform=transform, 
            cache_system=cache_system,
            world_size=world_size,
            rank=rank
        )
        
    def metadata(self):
        for idx in range(len(self.members)):
            data = self.get_data(idx)
            metadata = data["metadata"]
            yield metadata

    def get_members(self):
        if self.debug:
            members = [self.get_reader().next() for _ in range(5000)]
            return list(sorted(members, key=lambda m: m.name))   
        return list(sorted(self.get_reader().getmembers(), key=lambda m: m.name))       

    def get_item_from_archive(self, member: tarfile.TarInfo):
        """
        Implements a function to load a file from a TarFile object.

        :param member: The TarInfo object representing the file to load.

        :returns : The data loaded from the file.
        """
        # Loads file from TarFile stored as numpy array
        buffer = io.BytesIO()
        buffer.write(self.get_reader().extractfile(member).read())
        buffer.seek(0)
        data = np.load(buffer, allow_pickle=True)

        data = {key : values for key, values in data.items()}

        return data
        
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Implements the `__getitem__` method for the dataset.

        :param idx: The index of the item to retrieve.

        :returns : The item at the given index.
        """
        data = self.get_data(idx)
        
        img = data["image"] # assuming 'img' key
        metadata = data["metadata"].item()
        # if img.size != 224 * 224:
        #     print(img.shape)
        #     print(metadata)
        
        img = img / 255.

        if self.transform is not None:
            # This assumes that there is a conversion to torch Tensor in the given transform
            img = self.transform(img)
            if isinstance(img, list):
                img = [x.float() for x in img]
            else:
                img = img.float()
        else:
            if img.ndim == 2:
                img = img[np.newaxis]
            img = torch.tensor(img, dtype=torch.float32)

        if self.return_metadata:
            # Ensures all keys are not None
            metadata = ensure_values(metadata)
            return img, metadata
        return img # and whatever other metadata we like
    
def ensure_values(obj):
    """
    Ensures that the values from a dict match the requirements
    of the default collate function from torch
    """
    for key, value in obj.items():
        if isinstance(value, dict):
            # Recursively ensure values on dictionaries
            obj[key] = ensure_values(value)
        elif isinstance(value, (str, float, int)):
            # Converts strings to list of strings
            obj[key] = str(value)
        elif value is None:
            # Converts None values to list of strings
            obj[key] = "None"
    return obj

class HPADataset(ArchiveDataset):
    """
    Dataset class for loading and processing image data from a zip file.
    This dataset is designed for the HPA dataset.
    """
    def __init__(
            self,
            zip_path: str,
            use_cache: bool = False, 
            max_cache_size: int = 16e9, 
            image_channels: int = 1, 
            transform: Any = None, 
            cache_system=None, 
            return_metadata: bool=False,
            world_size: int = 1,
            rank: int = 0,
            *args, **kwargs
    ) -> None :
        """
        Instantiates a new ``HPADataset`` object.

        :param zip_path: The path to the zip file to load data from.
        :param transform: The transformation to apply to the image data.
        """
        super(HPADataset, self).__init__(
            zip_path, 
            use_cache=use_cache, 
            max_cache_size=max_cache_size,
            transform=transform,
            cache_system=cache_system,
            world_size=world_size,
            rank=rank
        )

        self.image_channels = image_channels
        self.return_metadata = return_metadata
    
    def get_members(self):

        file_list = []
        for f in self.get_reader().namelist():
            if f.endswith('.png'):
                file_list.append(f)
        return file_list
    
    def get_item_from_archive(self, member):
        """
        Implements a function to load a file from a ZipFile.

        :param member: The name of the object representing the file to load.

        :returns : The data loaded from the file.
        """
        data = self.get_reader().read(member)
        img = Image.open(io.BytesIO(data))
        img = numpy.array(img)
        return img

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Implements the ``__getitem__`` method for the dataset.

        :param idx: The index of the item to retrieve.

        :returns: The item at the given idex.
        """
        img = self.get_data(idx)

        # image = transforms.ToTensor()(img)
        m, M = img.min(), img.max()
        img = (img - m) / (M - m)

        if self.transform is not None:
            # This assumes that there is a conversion to torch Tensor in the given transform
            img = self.transform(img)
            if isinstance(img, list):
                img = [x.float() for x in img]
            else:
                img = img.float()
        else:
            if img.ndim == 2:
                img = img[np.newaxis]
            img = torch.tensor(img, dtype=torch.float32)  
        return img
