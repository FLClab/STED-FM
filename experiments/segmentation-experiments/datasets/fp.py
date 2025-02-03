
import numpy
import glob
import os
import torch
import random
import h5py
from typing import Tuple
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

TARGET_RES = 0.022724609375

def get_resolution(tif_fl, sh):
    with tifffile.TiffFile(tif_fl) as tif:
        if "XResolution" in tif.pages[0].tags:
            x1, x2 = tif.pages[0].tags["XResolution"].value
            if tif.pages[0].tags["ResolutionUnit"].value == 3: # RESUNIT.CENTIMETER
                x2 = x2 * 10000
            return (x2 / x1) * (tif.pages[0].shape[0] / sh)
        else:
            return TARGET_RES

class FPConfiguration(Configuration):

    num_classes: int = 2
    criterion: str = "MSELoss"
    min_annotated_ratio: float = 0.1

class FPDataset(Dataset):

    KEEP_CHANNEL = 0

    def __init__(self, path: str, files: Union[list, None] = None, transform = None, data_aug: float = 0, validation: bool = False, 
                 size :int = 256, step:float=0.75, cache_system:dict=None, n_channels:int=1, **kwargs) -> None:
        super(FPDataset, self).__init__()

        self.path = path
        self.files = files
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.size = size
        self.step = step
        self.validation = validation
        self.data_aug = data_aug
        self.n_channels = n_channels
        self.classes = ["FP", "SD"]

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
        uniques = numpy.unique(label)
        masks = []
        for unique in uniques[1:]:
            mask = label == unique
            masks.append(mask)
        # Makes sure only two masks in images
        if len(masks) > 2:
            sums = sorted([(i, mask.sum()) for i, mask in enumerate(masks)], key=lambda x: x[-1], reverse=True)
            masks = [masks[i] for i, _ in sums[:2]]
        return numpy.stack(masks, axis=0)

    def _rescale_image(self, file, image):
        """
        Rescales the image size to the target resolution

        :param image: A `numpy.ndarray` of the image

        :returns: A `numpy.ndarray` of the rescaled image
        """
        h, w = image.shape[-2], image.shape[-1]
        res = get_resolution(file, image.shape[-1])
        scale = round(res/TARGET_RES, 2)
        if scale != 1:
            newW, newH = int(scale * w), int(scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small'
            image = transform.resize(image, (newH, newW))
        return image
    
    def generate_valid_samples(self):
        samples = []

        if self.files is None:
            files = glob.glob(os.path.join(self.path, "images", "*.tif"))
        else:
            files = [os.path.join(self.path, "images", file) for file in self.files]

        for file in tqdm(files, desc="Files", leave=False):
            image = tifffile.imread(file)
            if image.ndim > 2:
                image = image[self.KEEP_CHANNEL]

            image = self._rescale_image(file, image)

            label_name = file.replace("images", "labels")
            label_name = label_name.replace(".tif", "_bin.tif")
            if os.path.isfile(label_name):
                try:
                    label = tifffile.imread(label_name)
                except tifffile.tifffile.TiffFileError:
                    print(f"Error reading {label_name}")
                    continue
                
                label = self._convert_to_masks(label)
                if len(label) > 2:
                    print(file, label_name)

                for j in range(0, image.shape[-2] - self.size, int(self.size * self.step)):
                    for i in range(0, image.shape[-1] - self.size, int(self.size * self.step)):
                        label_crop = numpy.sum(label[:, j : j + self.size, i : i + self.size], axis=0)
                        if numpy.any(label_crop):
                            sample = {
                                "cache-key" : file,
                                "image-name" : file,
                                "label-name" : label_name,
                                "slc" : (
                                    slice(j, j + self.size), slice(i, i + self.size)
                                ),
                            }
                            samples.append(sample)
                self.__cache[file] = (image, label)
            else:
                print(f"Label not found for {file}")
        return samples
    
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
        cache_key = sample["cache-key"]
        if cache_key in self.__cache:
            image, label = self.__cache[cache_key]
        else:
            image = tifffile.imread(sample["image-name"])
            if image.ndim > 2:
                image = image[self.KEEP_CHANNEL]
            image = self._rescale_image(sample["image-name"], image)
            label = tifffile.imread(sample["label-name"])
            label = self._convert_to_masks(label)

        slc = sample["slc"]
        image_crop, label_crop = image[slc], label[:, slc[0], slc[1]]

        if image_crop.size != self.size*self.size:
            image_crop = numpy.pad(image_crop, ((0, self.size - image_crop.shape[0]), (0, self.size - image_crop.shape[1])), "symmetric")
            label_crop = numpy.pad(label_crop, ((0, 0), (0, self.size - label_crop.shape[1]), (0, self.size - label_crop.shape[2])), "symmetric")

        image_crop = image_crop.astype(numpy.float32)
        label_crop = label_crop.astype(numpy.float32)

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
    cfg.dataset_cfg = FPConfiguration()

    if cfg.in_channels == 3:
        # ImageNet normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.0695771782959453, 0.0695771782959453, 0.0695771782959453], std=[0.12546228631005282, 0.12546228631005282, 0.12546228631005282])
            transforms.Normalize(mean=[0.16182324290275574, 0.16182324290275574, 0.16182324290275574], std=[0.10339891165494919, 0.10339891165494919, 0.10339891165494919])
        ])
    else:
        transform = transforms.ToTensor()    

    training_path = os.path.join(BASE_PATH, "segmentation-data", "footprocess")
    with open(os.path.join(training_path, "training.txt"), "r") as f:
        training_files = f.readlines()
        training_files = [f.strip() for f in training_files]
    
    validation_path = os.path.join(BASE_PATH, "segmentation-data", "footprocess")
    with open(os.path.join(training_path, "validation.txt"), "r") as f:
        validation_files = f.readlines()
        validation_files = [f.strip() for f in validation_files]

    testing_path = os.path.join(BASE_PATH, "segmentation-data", "footprocess", "test")

    if test_only:
        training_dataset, validation_dataset = None, None 
    else:
        training_dataset = FPDataset(
            path=training_path,
            transform=transform,
            files=training_files,
            data_aug=0.5,
            validation=False,
            size=224,
            step=0.75,
            n_channels=cfg.in_channels
        )
        validation_dataset = FPDataset(
            path=validation_path,
            transform=transform,
            files=validation_files,
            data_aug=0,
            validation=True,
            size=224,
            step=0.75,
            n_channels=cfg.in_channels
        )
    testing_dataset = FPDataset(
        path=testing_path,
        transform=transform,
        data_aug=0,
        validation=True,
        size=224,
        step=0.75,
        n_channels=cfg.in_channels,
        return_foregound=True
    )
    return training_dataset, validation_dataset, testing_dataset    

if __name__ == "__main__":

    import shutil
    from sklearn.model_selection import train_test_split
    images = glob.glob(os.path.join(BASE_PATH, "segmentation-data", "footprocess", "images", "*.tif"))
    X_train, X_test = train_test_split(images, test_size=0.2, random_state=42)

    with open(os.path.join(BASE_PATH, "segmentation-data", "footprocess", "training.txt"), "w") as f:
        for x in X_train:
            x = os.path.basename(x)
            f.write(f"{x}\n")
    with open(os.path.join(BASE_PATH, "segmentation-data", "footprocess", "validation.txt"), "w") as f:
        for x in X_test:
            x = os.path.basename(x)
            f.write(f"{x}\n")

    # Testing modification
    os.makedirs(os.path.join(BASE_PATH, "segmentation-data", "footprocess", "test", "labels"), exist_ok=True)
    images = glob.glob(os.path.join(BASE_PATH, "segmentation-data", "footprocess", "test", "images", "*.tif"))
    for image in images:
        label = image.replace("test/images", "labels")
        label = label.replace(".tif", "_bin.tif")
        if not os.path.isfile(label):
            print(image)
            continue 
        outdir = os.path.dirname(image)
        outdir = outdir.replace("images", "labels")
        shutil.copy(label, outdir)