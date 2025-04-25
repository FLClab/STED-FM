
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

from stedfm.DEFAULTS import BASE_PATH
from stedfm.configuration import Configuration

class LionessConfiguration(Configuration):

    num_classes: int = 2
    criterion: str = "MSELoss"
    min_annotated_ratio: float = 0.1

def convert_to_semantic_segmentation(label, tickness=1):
    """
    Converts the label to a semantic segmentation format
    """
    converted = []
    for frame in label:
        # Convert label to semantic segmentation mask
        label_mask = numpy.zeros_like(frame)
        label_mask[frame > 0] = 1

        # Generate boundary mask
        boundary_mask = numpy.zeros_like(frame)
        boundary_mask[1:, :] = numpy.logical_or(boundary_mask[1:, :], frame[:-1, :] != frame[1:, :])
        boundary_mask[:-1, :] = numpy.logical_or(boundary_mask[:-1, :], frame[1:, :] != frame[:-1, :])
        boundary_mask[:, 1:] = numpy.logical_or(boundary_mask[:, 1:], frame[:, :-1] != frame[:, 1:])
        boundary_mask[:, :-1] = numpy.logical_or(boundary_mask[:, :-1], frame[:, 1:] != frame[:, :-1])

        # Make boundaries 3 pixel thick
        boundary_mask = morphology.binary_dilation(boundary_mask, morphology.disk(tickness))

        # Combine object mask and boundary mask
        segmentation_mask = numpy.stack([label_mask, boundary_mask], axis=0)    
        converted.append(segmentation_mask)    
    return numpy.array(converted)    

class LionessDataset(Dataset):
    """
    A `Dataset` class for the Lioness dataset. This class is used to load the dataset
    from the HDF5 files.
    """
    def __init__(self, path, transform=None, data_aug=0, validation=False, size=256, step=0.75, cache_system=None, n_channels=1, return_foregound=False, **kwargs):
        super(LionessDataset, self).__init__()

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
        self.classes = ["Cell", "Boundary"]

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

        image_names = glob.glob(os.path.join(self.path, "*.h5"))
        label_names = [file.replace("img", "labels").replace(".h5", ".tif") for file in image_names]
        for image_name, label_name in zip(image_names, label_names):
            label = tifffile.imread(label_name)
            label = convert_to_semantic_segmentation(label)

            with h5py.File(image_name, "r") as image_file:
                volume = image_file["main"][()]
                for k, frame in enumerate(volume):
                    for j in range(0, frame.shape[-2], int(self.size * self.step)):
                        for i in range(0, frame.shape[-1], int(self.size * self.step)):
                            slc = tuple([slice(k, k + 1), slice(j, j + self.size), slice(i, i + self.size)])
                            samples.append({
                                "image-name" : image_name,
                                "group-name" : "main",
                                "frame-idx" : k,
                                "slc" : slc,
                                "position" : (j, i)
                            })
            self.cache[image_name] = {"data" : volume, "label" : label}
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
        image_crop = self.cache[sample["image-name"]]["data"][sample["slc"]][0] # Keeps single frame
        label_crop = self.cache[sample["image-name"]]["label"][sample["slc"][0], :, sample["slc"][1], sample["slc"][2]][0]

        # Normalizes the image
        image_crop = (image_crop - image_crop.min()) / (image_crop.max() - image_crop.min())
    
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

class TestingLionessDataset(Dataset):
    """
    Testing dataset of the Lioness dataset

    This dataset assumes that the image is a 3D volume with two channels. The first channel
    contains the ground truth. The ground truth in this dataset corresponds in a single structure
    that needs to be segmented from the image. The second channel contains the image that needs
    to be segmented.
    """
    def __init__(self, path, transform=None, size=224, step=0.75, out_channels=2, return_foregound=False, cache_system=None, **kwargs):

        self.path = path
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.size = size
        self.step = step
        self.out_channels = out_channels
        self.return_foregound = return_foregound
        self.classes = ["Cell", "Boundary"]

        self.cache = {}
        if cache_system is not None:
            self.cache = cache_system

        self.samples = self.generate_valid_samples()

    def generate_valid_samples(self):
        """
        Generates a list of valid samples from the dataset. This is performed only
        once when the dataset is created.
        """
        samples = []

        image_names = glob.glob(os.path.join(self.path, "*.tif"))
        for image_name in image_names:
            volume = tifffile.imread(image_name)
            label, frame = volume[:, 0], volume[:, 1]

            # Threshold gfp channel
            threshold = filters.threshold_otsu(label)
            foreground = label > threshold
            label = convert_to_semantic_segmentation(foreground)

            if self.return_foregound:
                label = numpy.concatenate((label, foreground[:, numpy.newaxis]), axis=1)

            for k, frame in enumerate(volume):
                for j in range(0, frame.shape[-2], int(self.size * self.step)):
                    for i in range(0, frame.shape[-1], int(self.size * self.step)):
                        slc = tuple([slice(k, k + 1), slice(j, j + self.size), slice(i, i + self.size)])
                        samples.append({
                            "image-name" : image_name,
                            "group-name" : "main",
                            "frame-idx" : k,
                            "slc" : slc,
                            "position" : (j, i)
                        })
            self.cache[image_name] = {"data" : volume[:, 1], "label" : label}
        return samples      

    def __getitem__(self, index : int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implements the `__getitem__` method for the `Dataset` class

        :param index : An `int` of the index of the sample to retrieve

        :returns : A tuple of the input and output tensors
        """
        sample = self.samples[index]
        image_crop = self.cache[sample["image-name"]]["data"][sample["slc"]][0] # Keeps single frame
        label_crop = self.cache[sample["image-name"]]["label"][sample["slc"][0], :, sample["slc"][1], sample["slc"][2]][0]

        # Normalizes the image
        image_crop = (image_crop - image_crop.min()) / (image_crop.max() - image_crop.min())
    
        if image_crop.size != self.size*self.size:
            image_crop = numpy.pad(image_crop, ((0, self.size - image_crop.shape[0]), (0, self.size - image_crop.shape[1])), "symmetric")
            label_crop = numpy.pad(label_crop, ((0, 0), (0, self.size - label_crop.shape[1]), (0, self.size - label_crop.shape[2])), "symmetric")

        image_crop = image_crop.astype(numpy.float32)
        label_crop = label_crop.astype(numpy.float32)

        if self.n_channels == 3:
            image_crop = numpy.tile(image_crop[numpy.newaxis], (3, 1, 1))
            image_crop = numpy.moveaxis(image_crop, 0, -1)
        img = self.transform(image_crop)
        y = torch.tensor(label_crop > 0, dtype=torch.float32)
        return x, y

    def __len__(self):
        """
        Implements the `len` method for the `Dataset` class

        :returns : An `int` of the number of samples in the dataset
        """        
        return len(self.samples)

def get_dataset(cfg : dataclass, test_only : bool = False, **kwargs) -> Tuple[Dataset, Dataset, Dataset]:

    # Updates the configuration inplace
    cfg.dataset_cfg = LionessConfiguration()

    if cfg.in_channels == 3:
        # ImageNet normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.0695771782959453, 0.0695771782959453, 0.0695771782959453], std=[0.12546228631005282, 0.12546228631005282, 0.12546228631005282])
            transforms.Normalize(mean=[0.6304965615272522, 0.6304965615272522, 0.6304965615272522], std=[0.2132348269224167, 0.2132348269224167, 0.2132348269224167])
        ])
    else:
        transform = transforms.ToTensor()    

    training_path = os.path.join(BASE_PATH, "segmentation-data", "lioness", "train")
    validation_path = os.path.join(BASE_PATH, "segmentation-data", "lioness", "valid")
    testing_path = os.path.join(BASE_PATH, "segmentation-data", "lioness", "test")

    if test_only:
        training_dataset, validation_dataset = None, None
    else:
        training_dataset = LionessDataset(
            path=training_path,
            transform=transform,
            data_aug=0.5,
            validation=False,
            size=224,
            step=0.75,
            n_channels=cfg.in_channels
        )
        validation_dataset = LionessDataset(
            path=validation_path,
            transform=transform,
            data_aug=0,
            validation=True,
            size=224,
            step=0.75,
            n_channels=cfg.in_channels
        )
    testing_dataset = LionessDataset(
        path = testing_path,
        transform=transform,
        validation=True,
        size = 224,
        step = 0.75,
        n_channels = cfg.in_channels,
        return_foregound=False
    )
    return training_dataset, validation_dataset, testing_dataset

if __name__ == "__main__":

    cfg = LionessConfiguration()
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
    
