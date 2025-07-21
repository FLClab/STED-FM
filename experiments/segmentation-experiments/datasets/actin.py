import numpy
import glob
import os
import torch
import random
import h5py
from typing import Tuple
from dataclasses import dataclass
from skimage import io
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms

from stedfm.DEFAULTS import BASE_PATH
from stedfm.configuration import Configuration

class FActinConfiguration(Configuration):

    num_classes: int = 2
    criterion: str = "MSELoss"
    min_annotated_ratio: float = 0.1

class HDF5Dataset(Dataset):
    """
    Creates a `Dataset` from a HDF5 file. It loads all the HDF5 file in cache. This
    increases the loading speed.

    :param file_path: A `str` to the hdf5 file
    :param data_aug: A `float` in range [0, 1]
    :param validation: (optional) Wheter the Dataset is for validation (no data augmentation)
    :param size: (optional) The size of the crops
    :param step: (optional) The step between each crops
    :param cache_system: (optional) A `dict` to store the cache
    :param out_channels: (optional) The number of output channels to return
    :param return_foregound: (optional) Wheter to return the foreground mask
    """
    def __init__(self, file_path, transform=None, data_aug=0, validation=False, size=256, step=0.75, cache_system=None, n_channels=1, return_foregound=False, return_index=False,**kwargs):
        super(HDF5Dataset, self).__init__()

        self.file_path = file_path
        self.return_index = return_index
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.size = size
        self.step = step
        self.validation = validation
        self.data_aug = data_aug
        self.n_channels = n_channels
        self.return_foregound = return_foregound
        self.classes = ["Rings", "Fibers"]

        self.cache = {}
        if cache_system is not None:
            self.cache = cache_system

        self.samples = self.generate_valid_samples()

    def generate_valid_samples(self):
        """
        Generates a list of valid samples from the dataset. This is performed only
        once at each training
        """
        samples = []
        stats = []
        with h5py.File(self.file_path, "r") as file:
            for group_name, group in tqdm(file.items(), desc="Groups", leave=False):
                data = group["data"][()].astype(numpy.float32) # Images
                label = group["label"][()] # shape is Rings, Fibers, and Dendrite
                shapes = group["label"].attrs["shapes"] # Not all images have same shape
                for k, (dendrite_mask, shape) in enumerate(zip(label[:, -1], shapes)):
                    for j in range(0, shape[0], int(self.size * self.step)):
                        for i in range(0, shape[1], int(self.size * self.step)):
                            dendrite = dendrite_mask[j : j + self.size, i : i + self.size]
                            dendrite = label[k, :2, j : j + self.size, i : i + self.size] > 0
                            if dendrite.sum() >= 0.1 * self.size * self.size: # dendrite is at least 10% of image
                                stats.append((numpy.mean(data[k, j : j + self.size, i : i + self.size]), numpy.std(data[k, j : j + self.size, i : i + self.size])))
                                samples.append((group_name, k, j, i))
                if self.return_foregound:
                    self.cache[group_name] = {"data" : data, "label" : label}
                else:
                    self.cache[group_name] = {"data" : data, "label" : label[:, :-1]}
        print(f"{numpy.mean(stats, axis=0)=}")
        return samples

    def __getitem__(self, index):
        """
        Implements the `__getitem__` function of the `Dataset`

        :param index: An `int` of the sample to return

        :returns: A `torch.tensor` of the image
                  A `torch.tensor` of the label
        """
        group_name, k, j, i = self.samples[index]

        image_crop = self.cache[group_name]["data"][k, j : j + self.size, i : i + self.size]
        label_crop = self.cache[group_name]["label"][k, :, j : j + self.size, i : i + self.size]

        if image_crop.size != self.size*self.size:
            image_crop = numpy.pad(image_crop, ((0, self.size - image_crop.shape[0]), (0, self.size - image_crop.shape[1])), "constant")
            label_crop = numpy.pad(label_crop, ((0, 0), (0, self.size - label_crop.shape[1]), (0, self.size - label_crop.shape[2])), "constant")

        image_crop = image_crop.astype(numpy.float32)
        label_crop = label_crop.astype(numpy.float32)

        # Applies data augmentation
        if not self.validation:

            if random.random() < self.data_aug:
                # random rotation 90
                number_rotations = random.randint(1, 3)
                image_crop = numpy.rot90(image_crop, k=number_rotations).copy()
                label_crop = numpy.array([numpy.rot90(l, k=number_rotations).copy() for l in label_crop])

            if random.random() < self.data_aug:
                # left-right flip
                image_crop = numpy.fliplr(image_crop).copy()
                label_crop = numpy.array([numpy.fliplr(l).copy() for l in label_crop])

            if random.random() < self.data_aug:
                # up-down flip
                image_crop = numpy.flipud(image_crop).copy()
                label_crop = numpy.array([numpy.flipud(l).copy() for l in label_crop])

            if random.random() < self.data_aug:
                # intensity scale
                intensityScale = numpy.clip(numpy.random.lognormal(0.01, numpy.sqrt(0.01)), 0, 1)
                image_crop = numpy.clip(image_crop * intensityScale, 0, 1)

            if random.random() < self.data_aug:
                # gamma adaptation
                gamma = numpy.clip(numpy.random.lognormal(0.005, numpy.sqrt(0.005)), 0, 1)
                image_crop = numpy.clip(image_crop**gamma, 0, 1)

        if self.n_channels == 3:
            image_crop = numpy.tile(image_crop[numpy.newaxis], (3, 1, 1))
            image_crop = numpy.moveaxis(image_crop, 0, -1)
        img = self.transform(image_crop)
        mask = torch.tensor(label_crop > 0, dtype=torch.float32)
        if self.return_index:
            return img, mask, {"dataset-idx": index}
        else:
            return img, mask

    def __len__(self):
        return len(self.samples)
    
def get_dataset(cfg:dataclass, test_only:bool=False, **kwargs) -> Tuple[Dataset, Dataset, Dataset]:

    # Updates the configuration inplace
    cfg.dataset_cfg = FActinConfiguration()

    if cfg.in_channels == 3:
        # ImageNet normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.0695771782959453, 0.0695771782959453, 0.0695771782959453], std=[0.12546228631005282, 0.12546228631005282, 0.12546228631005282])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transforms.Normalize(mean=[0.023, 0.023, 0.023], std=[0.027, 0.027, 0.027])
        ])
    else:
        transform = transforms.ToTensor()        

    hdf5_training_path = os.path.join(BASE_PATH, "segmentation-data", "factin", "training_small-dataset_20240418.hdf5")
    hdf5_validation_path = os.path.join(BASE_PATH, "segmentation-data", "factin", "validation_small-dataset_20240418.hdf5")
    hdf5_testing_path = os.path.join(BASE_PATH, "segmentation-data", "factin", "testing_small-dataset_20240418.hdf5")

    if test_only:
        training_dataset, validation_dataset = None, None
    else:
        training_dataset = HDF5Dataset(
            file_path=hdf5_training_path,
            transform=transform,
            data_aug=0.5,
            validation=False,
            size=224,
            step=0.75,
            n_channels=cfg.in_channels,
            **kwargs
        )
        validation_dataset = HDF5Dataset(
            file_path=hdf5_validation_path,
            transform=transform,
            data_aug=0,
            validation=True,
            size=224,
            step=0.75,
            n_channels=cfg.in_channels,
            **kwargs
        )
    testing_dataset = HDF5Dataset(
        file_path=hdf5_testing_path,
        transform=transform,
        data_aug=0,
        validation=True,
        size=224,
        step=0.75,
        n_channels=cfg.in_channels,
        return_foregound=True,
        return_index=kwargs.get("return_index", False),
        **kwargs
    )
    return training_dataset, validation_dataset, testing_dataset