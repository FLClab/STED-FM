
import h5py
import json
import random
import numpy
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple
from tqdm import tqdm

class SemanticProteinSegmentationDataset(Dataset):

    ROUND = 0
    ELONGATED = 1
    PERFORATED = 2
    MULTIDOMAINS = 3

    def __init__(self, h5file: str, transform=None, n_channels=1, validation=False, data_aug=0.5) -> None:
        self.h5file = h5file

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

        self.n_channels = n_channels
        self.validation = validation
        self.data_aug = data_aug

        self.image_size = 224

        self.classes = ['round', 'elongated', 'perforated', 'multidomains']

        with h5py.File(self.h5file, "r") as file:
            
            valid_indices = []
            for c in self.classes:
                indices = numpy.argwhere(file["labels"][()].ravel() == getattr(self, c.upper())).ravel().tolist()
                valid_indices.extend(indices)

            metadata = file["crops"].attrs["metadata"]
            metadata = json.loads(metadata)
            for i in range(len(metadata)):
                metadata[i]["idx"] = i
            self.metadata = [metadata[idx] for idx in valid_indices]

            self.images = {
                key : {"image" : value[()], 
                       "label" : numpy.zeros((value.shape[0], len(self.classes), value.shape[1], value.shape[2]), dtype=numpy.float32)} 
                for key, value in file["original_images"].items()
            }   
        
        self.create_segmentation()
        self.valid_indices = self.get_valid_indices()
    
    def create_segmentation(self):
        for i, metadata in enumerate(tqdm(self.metadata)):

            label = metadata["class_majority"]

            idx = metadata["idx"]
            with h5py.File(self.h5file, "r") as file:
                mask = file["masks"][idx][()]
            
            votes = numpy.array(metadata["seg_votes"])
            votes = votes[votes < len(mask)]
            masks = mask[votes]
            if len(votes) < 1:
                continue
            mask = numpy.mean(masks, axis=0)
            mask = mask.astype(numpy.float32)

            image = self.images[metadata["img_name"]]["label"][metadata["channel"]][0]
            coord = metadata["id_coord"]
            slc = tuple(
                slice(max(0, coord[i] - 40), min(coord[i] + 40, image.shape[i]))
                for i in range(len(coord))
            )
            self.images[metadata["img_name"]]["label"][metadata["channel"]][label][slc] += mask

    def get_valid_indices(self):
        valid_indices = []
        for key, values in self.images.items():
            img = values["image"]
            label = values["label"]
            for chan in range(img.shape[0]):
                for j in range(0, img.shape[-2], self.image_size):
                    for i in range(0, img.shape[-1], self.image_size):
                        if numpy.sum(label[chan, :, j:j+self.image_size, i:i+self.image_size]) > 0:
                            valid_indices.append((key, chan, j, i))
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.Tensor]:
        
        key, chan, j, i = self.valid_indices[idx]
        
        image_crop = self.images[key]["image"][chan, j:j+self.image_size, i:i+self.image_size]
        label_crop = self.images[key]["label"][chan, :, j:j+self.image_size, i:i+self.image_size]

        # Normalize image
        image_crop = (image_crop - image_crop.min()) / (image_crop.max() - image_crop.min())
        image_crop = image_crop.astype(numpy.float32)
        label_crop = (label_crop - label_crop.min()) / (label_crop.max() - label_crop.min())
        label_crop = label_crop.astype(numpy.float32)

        if image_crop.size != self.image_size ** 2:
            image_crop = numpy.pad(image_crop, ((0, self.image_size - image_crop.shape[-2]), (0, self.image_size - image_crop.shape[-1])), mode="symmetric")
            label_crop = numpy.pad(label_crop, ((0, 0), (0, self.image_size - label_crop.shape[-2]), (0, self.image_size - label_crop.shape[-1])), mode="symmetric")
        
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
        mask = torch.tensor(label_crop)

        return img, mask

def main():
    dataset = SemanticProteinSegmentationDataset(
        h5file="path/to/your/dataset.h5",
        n_channels=1,
        validation=False,
        data_aug=0.5
    )

    for img, mask in dataset:
        print(img.shape, mask.shape)
    
if __name__ == "__main__":
    main()