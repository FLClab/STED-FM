import tarfile 
import numpy
import numpy as np
import io
import torch
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
import copy
from skimage import filters

from DEFAULTS import BASE_PATH

LOCAL_CACHE = {}

def get_dataset(name: str, path: str, **kwargs):
    if name == "CTC":
        dataset = CTCDataset(path, **kwargs)
    elif name == "JUMP":
        dataset = JUMPCPDataset(h5file=path, **kwargs)
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
        self.labels = []
        for i, class_name in enumerate(classes):
            class_folder = os.path.join(data_folder, class_name)
            self.class_files[class_name] = self._filter_files(class_folder)
            self.samples[class_name] = self._get_sampled_files(self.class_files[class_name], self.num_samples.get(class_name))
            self.labels.extend([i] * len(self.samples[class_name]))



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

class NeuralActivityStates(Dataset):
    def __init__(
        self, 
        h5file: str,
        transform: Callable = None,
        n_channels: int = 1,
        num_samples: int = None,
        num_classes: int = 4,
        protein_id: int = 3
    ) -> None:
        self.h5file = h5file 
        self.transform = transform 
        self.n_channels = n_channels 
        self.num_samples = num_samples 
        self.num_classes = num_classes 
        if self.num_samples is None:
            with h5py.File(h5file, "r") as handle:
                protein_ids = np.where(handle["proteins"][()] == protein_id)
                labels = handle["conditions"][protein_ids]
                images = handle["images"][protein_ids]
                proteins = handle["proteins"][protein_ids]
                self.labels, indices = self.__get_sample_ids(labels, method="max-TTX")
               
                indices = indices.astype(np.int64)
                
                self.images = images[indices]
                self.proteins = proteins[indices]
                self.dataset_size = self.labels.shape[0]
                print(self.proteins.shape, self.images.shape, self.labels.shape, self.dataset_size)
                
        else:
            raise NotImplementedError("Subset sampler not implemented yet for this dataset.")
        

    def __get_sample_ids(self, labels: np.ndarray, method: str = "merge-TTX") -> np.ndarray:
        indices = []
        new_labels = []
        if method == "merge-TTX":
            for i, l in enumerate(labels):
                if l <= 3:
                    indices.append(i)
                    new_labels.append(l)
                elif 4 <= l <= 6:
                    indices.append(i)
                    new_labels.append(3) # Not 4 b/c KCL (=3) was already removed
        elif method == "max-TTX":
            for i, l in enumerate(labels):
                if l <= 3:
                    indices.append(i)
                    new_labels.append(l)
                elif l == 6:
                    indices.append(i)
                    new_labels.append(3)
        else:
            raise ValueError(f"Method `{method}` for resetting labels is not implemented yet.")

        return np.array(new_labels), np.array(indices)

    def __len__(self):
        return self.dataset_size 
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        with h5py.File(self.h5file, "r") as handle:
            img = self.images[idx]
            label = self.labels[idx]
        
        if self.n_channels == 3:
            img = np.tile(img[np.newaxis], (3, 1, 1))
            img = np.moveaxis(img, 0, -1)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize(mean=[0.0695771782959453, 0.0695771782959453, 0.0695771782959453], std=[0.12546228631005282, 0.12546228631005282, 0.12546228631005282])(img)
        else:
            img = transforms.ToTensor()(img)

        return img, {"label": label, "protein": self.proteins[idx]}

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


class TarFLCDataset(Dataset):
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
                 rank: int = 0) -> None:
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
        self.__cache = {}
        self.__max_cache_size = max_cache_size
        self.tar_path = tar_path
        self.image_channels = image_channels
        self.transform = transform
        self.return_metadata = return_metadata

        # Multiprocessing settings for multi-gpu training
        self.world_size = world_size
        self.rank = rank

        worker = get_worker_info()
        worker = worker.id if worker else None
        self.tar_obj = {worker: tarfile.open(self.tar_path, "r")}

        # store headers of all files and folders by name
        members = list(sorted(self.tar_obj[worker].getmembers(), key=lambda m: m.name))
        # members = [self.tar_obj[worker].next() for _ in range(10000)]
        # members = list(self.tar_obj[worker].getmembers())
        self.members = self.__setup_multiprocessing(members)

        if use_cache and self.__max_cache_size > 0:
            self.__cache_size = 0
            if not cache_system is None:
                self.__cache = cache_system
            self.__fill_cache()

    def __setup_multiprocessing(self, members : list):
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

    def __get_item_from_tar(self, member: tarfile.TarInfo):
        """
        Implements a function to load a file from a TarFile object.

        :param member: The TarInfo object representing the file to load.

        :returns : The data loaded from the file.
        """
        # ensure a unique file handle per worker, in multiprocessing settings
        worker = get_worker_info()
        worker = worker.id if worker else None
        if worker not in self.tar_obj:
            self.tar_obj[worker] = tarfile.open(self.tar_path, "r")

        # Loads file from TarFile stored as numpy array
        buffer = io.BytesIO()
        buffer.write(self.tar_obj[worker].extractfile(member).read())
        buffer.seek(0)
        data = np.load(buffer, allow_pickle=True)

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
        """
        Implements the `__len__` method for the dataset.
        """
        return len(self.members)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Implements the `__getitem__` method for the dataset.

        :param idx: The index of the item to retrieve.

        :returns : The item at the given index.
        """
        if idx in self.__cache:
            data = self.__cache[idx]
        else:
            data = self.__get_item_from_tar(self.members[idx])
        
        img = data["image"] # assuming 'img' key
        metadata = data["metadata"]
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
            img = img[np.newaxis]
            img = torch.tensor(img, dtype=torch.float32)

        if self.return_metadata:
            return img, metadata
        return img # and whatever other metadata we like
    
    def __del__(self):
        """
        Close the TarFile file handles on exit.
        """
        for o in self.tar_obj.values():
            o.close()
            
    def __getstate__(self) -> dict:
        """
        Serialize without the TarFile references, for multiprocessing compatibility.
        """
        state = dict(self.__dict__)
        state['tar_obj'] = {}
        return state