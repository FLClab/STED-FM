import numpy
import glob
import os
import torch
import random
import h5py

from dataclasses import dataclass
from skimage import io
from torch.utils.data import Dataset
from tqdm import tqdm

import sys
sys.path.insert(0, "..")
from DEFAULTS import BASE_PATH

@dataclass
class FActinConfiguration:

    num_classes: int = 2
    criterion: str = "MSELoss"

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
    def __init__(self, file_path, data_aug=0, validation=False, size=256, step=0.75, cache_system=None, out_channels=1, return_foregound=False, **kwargs):
        super(HDF5Dataset, self).__init__()

        self.file_path = file_path
        self.size = size
        self.step = step
        self.validation = validation
        self.data_aug = data_aug
        self.out_channels = out_channels
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
        with h5py.File(self.file_path, "r") as file:
            for group_name, group in tqdm(file.items(), desc="Groups", leave=False):
                data = group["data"][()].astype(numpy.float32) # Images
                label = group["label"][()] # shape is Rings, Fibers, and Dendrite
                shapes = group["label"].attrs["shapes"] # Not all images have same shape
                for k, (dendrite_mask, shape) in enumerate(zip(label[:, -1], shapes)):
                    for j in range(0, shape[0], int(self.size * self.step)):
                        for i in range(0, shape[1], int(self.size * self.step)):
                            dendrite = dendrite_mask[j : j + self.size, i : i + self.size]
                            if dendrite.sum() >= 0.1 * self.size * self.size: # dendrite is at least 1% of image
                                samples.append((group_name, k, j, i))
                if self.return_foregound:
                    self.cache[group_name] = {"data" : data, "label" : label}
                else:
                    self.cache[group_name] = {"data" : data, "label" : label[:, :-1]}
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

        x = torch.tensor(image_crop, dtype=torch.float32)
        if self.out_channels > 1:
            x = x.unsqueeze(0)
            x = torch.tile(x, (self.out_channels, 1, 1))
        y = torch.tensor(label_crop > 0, dtype=torch.float32)
        return x, y

    def __len__(self):
        return len(self.samples)
    
def get_dataset(cfg:dataclass, test_only:bool=False, **kwargs) -> tuple[Dataset, Dataset, Dataset]:

    # Updates the configuration inplace
    cfg.dataset_cfg = FActinConfiguration()

    hdf5_training_path = os.path.join(BASE_PATH, "segmentation-data", "factin", "training_small-dataset_20240418.hdf5")
    hdf5_validation_path = os.path.join(BASE_PATH, "segmentation-data", "factin", "validation_small-dataset_20240418.hdf5")
    hdf5_testing_path = os.path.join(BASE_PATH, "segmentation-data", "factin", "testing_EXP192-block-glugly.hdf5")

    if test_only:
        training_dataset, validation_dataset = None, None
    else:
        training_dataset = HDF5Dataset(
            file_path=hdf5_training_path,
            data_aug=0.5,
            validation=False,
            size=256,
            step=0.75,
            out_channels=cfg.in_channels
        )
        validation_dataset = HDF5Dataset(
            file_path=hdf5_validation_path,
            data_aug=0,
            validation=True,
            size=256,
            step=0.75,
            out_channels=cfg.in_channels
        )
    testing_dataset = HDF5Dataset(
        file_path=hdf5_testing_path,
        data_aug=0,
        validation=True,
        size=256,
        step=0.75,
        out_channels=cfg.in_channels,
        return_foregound=True
    )
    return training_dataset, validation_dataset, testing_dataset