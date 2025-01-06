
import numpy
import glob
import os
import torch
import random
import h5py
from typing import Tuple, List
import tifffile
from dataclasses import dataclass
from skimage import io, transform
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Union
from torchvision import transforms

import sys
sys.path.insert(0, "..")
from DEFAULTS import BASE_PATH
from configuration import Configuration

class MitochondriaConfiguration(Configuration):

    num_classes: int = 1
    criterion: str = "MSELoss"
    min_annotated_ratio: float = 0.01

class MitochondriaDataset(Dataset):

    KEEP_CHANNEL = 0

    def __init__(
            self, 
            image_path: str, 
            label_path: str, 
            indices: Union[list, None] = None, 
            transform = None, 
            data_aug: float = 0, 
            validation: bool = False, 
            size :int = 256, 
            step:float=0.75, 
            cache_system:dict=None, 
            n_channels:int=1, **kwargs) -> None:
        super(MitochondriaDataset, self).__init__()

        self.image_path = image_path
        self.label_path = label_path
        self.indices = indices
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.size = size
        self.step = step
        self.validation = validation
        self.data_aug = data_aug
        self.n_channels = n_channels
        self.classes = ["fission"]

        self.__cache = {}
        if cache_system is not None:
            self.__cache = cache_system

        self.samples = self.generate_valid_samples()

    def _convert_to_masks(self, label):
        """
        Converts the label to multiple masks for semantic segmentation

        :param label: A `numpy.ndarray` of the label

        :returns: A `numpy.ndarray` of the masks
        """
        return label > 0

    def _rescale(self, image):
        """
        Rescales the image size to the target resolution

        :param image: A `numpy.ndarray` of the image

        :returns: A `numpy.ndarray` of the rescaled image
        """
        h, w = image.shape[-2], image.shape[-1]
        images = transform.resize(image, (self.size, self.size))
        return image
    
    def generate_valid_samples(self):
        samples = []

        with h5py.File(self.image_path, "r") as file:
            images = file["Mito"][self.indices]
            images = images / 255.0
        # print(numpy.mean(numpy.mean(images, axis=(-2, -1))), numpy.mean(numpy.std(images, axis=(-2, -1))))
        with h5py.File(self.label_path, "r") as file:
            labels = file["Proc"][self.indices]
            labels = labels / 255.0

        self.__cache["images"] = images
        self.__cache["labels"] = labels

        return numpy.arange(len(images))
    
    def __len__(self):
        """
        Implements the `len` method

        :returns : The number of samples
        """
        return len(self.samples)
    
    def __getitem__(self, index : int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implements the `__getitem__` function of the `Dataset`

        :param index: An `int` of the sample to return

        :returns: A `torch.tensor` of the image
                  A `torch.tensor` of the label
        """
        sample = self.samples[index]
        
        image = self.__cache["images"][sample]
        label = self.__cache["labels"][sample]

        image_crop = self._rescale(image)
        label_crop = self._rescale(label)
        label_crop = self._convert_to_masks(label_crop)
        
        image_crop = image_crop.astype(numpy.float32)
        label_crop = label_crop.astype(numpy.float32)

        if label_crop.ndim == 2:
            label_crop = label_crop[numpy.newaxis]

        image_crop = (image_crop - image_crop.min()) / (image_crop.max() - image_crop.min())

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
        return img, mask             
    
def get_dataset(cfg:dataclass, test_only:bool=False, **kwargs) -> Tuple[Dataset, Dataset, Dataset]:

    # Updates the configuration inplace
    cfg.dataset_cfg = MitochondriaConfiguration()

    if cfg.in_channels == 3:
        # ImageNet normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.080, 0.080, 0.080], std=[0.164, 0.164, 0.164])
        ])
    else:
        transform = transforms.ToTensor()    

    image_path = os.path.join(BASE_PATH, "segmentation-data", "mitochondria", "training_data", "Mito.h5")    
    label_path = os.path.join(BASE_PATH, "segmentation-data", "mitochondria", "training_data", "Proc.h5")
    
    training_indices = numpy.arange(0, 24000)
    validation_indices = numpy.arange(24000, 30000)
    testing_indices = numpy.arange(24000, 37000)

    if test_only:
        training_dataset, validation_dataset = None, None 
    else:
        training_dataset = MitochondriaDataset(
            image_path=image_path,
            label_path=label_path,
            transform=transform,
            indices=training_indices,
            data_aug=0.5,
            validation=False,
            size=224,
            step=0.75,
            n_channels=cfg.in_channels
        )
        validation_dataset = MitochondriaDataset(
            image_path=image_path,
            label_path=label_path,
            transform=transform,
            indices=validation_indices,
            data_aug=0,
            validation=True,
            size=224,
            step=0.75,
            n_channels=cfg.in_channels
        )
    testing_dataset = MitochondriaDataset(
        image_path=image_path,
        label_path=label_path,
        indices=testing_indices,
        data_aug=0,
        validation=True,
        size=224,
        step=0.75,
        n_channels=cfg.in_channels,
        return_foregound=True
    )
    return training_dataset, validation_dataset, testing_dataset    
