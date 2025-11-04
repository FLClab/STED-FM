
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

from stedfm.DEFAULTS import BASE_PATH
from stedfm.configuration import Configuration

class LCNConfiguration(Configuration):

    num_classes: int = 3
    criterion: str = "MSELoss"
    min_annotated_ratio: float = 0.1

class LCNDataset(Dataset):
    """
    A `Dataset` class for the LCN dataset. This class is used to load the dataset
    from the HDF5 files.
    """
    def __init__(self, path, transform=None, data_aug=0, validation=False, size=256, step=0.75, cache_system=None, n_channels=1, return_foregound=False, **kwargs):
        super(LCNDataset, self).__init__()

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
        self.classes = ["Canaliculi", "Lacunae", "Canals (Blood Vessels)"]

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
        image_names = list(filter(lambda x: "_mask" not in x, image_names))

        label_names = [file.replace(".tif", "_mask.tif") for file in image_names]

        statistics = defaultdict(list)
        for image_name, label_name in zip(image_names, label_names):

            image = tifffile.imread(image_name)
            if not os.path.exists(label_name):
                continue

            label = tifffile.imread(label_name)

            for j in range(0, image.shape[-2] - self.size, int(self.size * self.step)):
                for i in range(0, image.shape[-1] - self.size, int(self.size * self.step)):
                    slc = tuple([slice(j, j + self.size), slice(i, i + self.size)])
                    samples.append({
                        "image-name" : image_name,
                        "slc" : slc,
                        "position" : (j, i)
                    })

            statistics["mean"].append(numpy.mean(image))
            statistics["std"].append(numpy.std(image))
            self.cache[image_name] = {"data" : image, "label" : label}

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
        image_crop = self.cache[sample["image-name"]]["data"][sample["slc"]] # Keeps single frame
        label_crop = self.cache[sample["image-name"]]["label"][sample["slc"]]

        masks = []
        for i in range(len(self.classes)):
            masks.append((label_crop == (i + 1)).astype(numpy.float32))
        label_crop = numpy.stack(masks, axis=0)
    
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
    cfg.dataset_cfg = LCNConfiguration()

    if cfg.in_channels == 3:
        # ImageNet normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.0695771782959453, 0.0695771782959453, 0.0695771782959453], std=[0.12546228631005282, 0.12546228631005282, 0.12546228631005282])
            transforms.Normalize(mean=[0.10, 0.10, 0.10], std=[0.14, 0.14, 0.14])
        ])
    else:
        transform = transforms.ToTensor()    

    training_path = os.path.join(BASE_PATH, "segmentation-data", "lcn", "training")
    validation_path = os.path.join(BASE_PATH, "segmentation-data", "lcn", "validation")
    testing_path = os.path.join(BASE_PATH, "segmentation-data", "lcn", "testing")

    if test_only:
        training_dataset, validation_dataset = None, None
    else:
        training_dataset = LCNDataset(
            path=training_path,
            transform=transform,
            data_aug=0.5,
            validation=False,
            size=224,
            step=1.0,
            n_channels=cfg.in_channels
        )
        validation_dataset = LCNDataset(
            path=validation_path,
            transform=transform,
            data_aug=0,
            validation=True,
            size=224,
            step=1.0,
            n_channels=cfg.in_channels
        )
    testing_dataset = LCNDataset(
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

    cfg = LCNConfiguration()
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
    
