
import numpy
import glob
import os
import torch
import random
import h5py
import tifffile
from typing import Tuple
from dataclasses import dataclass
from skimage import morphology, filters
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms
from collections import defaultdict
from skimage.filters import threshold_otsu

from stedfm.DEFAULTS import BASE_PATH
from stedfm.configuration import Configuration

class DEEPD3Configuration(Configuration):

    num_classes: int = 2
    criterion: str = "MSELoss"
    min_annotated_ratio: float = 0.1

class DEEPD3Dataset(Dataset):
    """
    A `Dataset` class for the DEEPD3 dataset. This class is used to load the dataset
    from the HDF5 files.
    """
    def __init__(self, path, transform=None, data_aug=0, validation=False, size=256, step=0.75, cache_system=None, n_channels=1, return_foregound=False, **kwargs):
        super(DEEPD3Dataset, self).__init__()

        self.path = path

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
        self.classes = ["Dendrite", "Spine"]

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

        image_names = glob.glob(os.path.join(self.path, "*.tif"))
        statistics = defaultdict(list)
        for image_name in image_names:
            image = tifffile.imread(image_name)

            image, label = image[:, 0], image[:, 1:]

            m, M = numpy.min(image, axis=(-2, -1), keepdims=True), numpy.max(image, axis=(-2, -1), keepdims=True)
            image = (image - m) / (M - m + 1e-8)

            foreground_stack = image > threshold_otsu(image)
            for k in range(image.shape[0]):
                foreground = foreground_stack[k]
                for j in range(0, max(1, image.shape[-2] - self.size), int(self.size * self.step)):
                    for i in range(0, max(1, image.shape[-1] - self.size), int(self.size * self.step)):
                        slc = tuple([slice(j, j + self.size), slice(i, i + self.size)])

                        # Remove samples with less than 1% foreground
                        if numpy.count_nonzero(foreground[slc]) / (self.size * self.size) < 0.01:
                            continue

                        samples.append({
                            "image-name" : image_name,
                            "frame-idx": k,
                            "slc" : slc,
                            "position" : (j, i)
                        })
            self.cache[image_name] = {"data" : image, "label" : label}

            statistics["mean"].extend(numpy.mean(image, axis=(-2, -1)))
            statistics["std"].extend(numpy.std(image, axis=(-2, -1)))

        for key, value in statistics.items():
            print(f"{key}: {numpy.mean(value)}")
        
        return samples
    
    def __len__(self):
        """
        Implements the `len` method for the `Dataset` class

        :returns : An `int` of the number of samples in the dataset
        """
        return len(self.samples)
    
    def __getitem__(self, index : int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implements the `__getitem__` method for the `Dataset` class

        :param index : An `int` of the index of the sample to retrieve

        :returns : A tuple of the input and output tensors
        """
        sample = self.samples[index]
        image_crop = self.cache[sample["image-name"]]["data"][sample["frame-idx"]][sample["slc"]]
        label_crop = self.cache[sample["image-name"]]["label"][sample["frame-idx"]][:, sample["slc"][0], sample["slc"][1]]

        if image_crop.size != self.size*self.size:
            image_crop = numpy.pad(image_crop, ((0, self.size - image_crop.shape[0]), (0, self.size - image_crop.shape[1])), "symmetric")
            label_crop = numpy.pad(label_crop, ((0, 0), (0, self.size - label_crop.shape[1]), (0, self.size - label_crop.shape[2])), "symmetric")

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
        return img, mask

def get_dataset(cfg : dataclass, test_only : bool = False, **kwargs) -> Tuple[Dataset, Dataset, Dataset]:

    # Updates the configuration inplace
    cfg.dataset_cfg = DEEPD3Configuration()

    if cfg.in_channels == 3:
        # ImageNet normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.07, 0.07, 0.07], std=[0.05, 0.05, 0.05])
        ])
    else:
        transform = transforms.ToTensor()    

    training_path = os.path.join(BASE_PATH, "segmentation-data", "deepd3", "DeepD3_Training")
    testing_path = os.path.join(BASE_PATH, "segmentation-data", "deepd3", "DeepD3_Validation")

    if test_only:
        training_dataset, validation_dataset = None, None
    else:
        training_dataset = DEEPD3Dataset(
            path=training_path,
            transform=transform,
            data_aug=0.5,
            validation=False,
            size=224,
            step=1.0,
            n_channels=cfg.in_channels
        )
        subset = int(0.1 * len(training_dataset))
        random_indices = random.sample(range(len(training_dataset)), subset)
        validation_dataset = torch.utils.data.Subset(training_dataset, random_indices)
        training_dataset = torch.utils.data.Subset(training_dataset, list(set(range(len(training_dataset))) - set(random_indices)))

    testing_dataset = DEEPD3Dataset(
        path = testing_path,
        transform=transform,
        validation=True,
        size = 224,
        step = 1.0,
        n_channels = cfg.in_channels,
        return_foregound=False
    )
    return training_dataset, validation_dataset, testing_dataset

if __name__ == "__main__":

    cfg = DEEPD3Configuration()
    cfg.in_channels = 1
    _, _, dataset = get_dataset(cfg, test_only=True)
    
    x, y = dataset[140]
    x = x.numpy()
    y = y.numpy()
    from matplotlib import pyplot
    fig, axes = pyplot.subplots(1, 3)
    axes[0].imshow(y[0])
    axes[1].imshow(y[1])
    axes[2].imshow(x)
    fig.savefig("./test.png")
    
