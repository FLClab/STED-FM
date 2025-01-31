import torch 
import os 
import h5py 
import numpy as np
import numpy
import json
import random
import ast
import io
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from typing import Tuple, Optional, Callable
from torchvision import transforms
from dataclasses import dataclass
from matplotlib import pyplot
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.registration import phase_cross_correlation
from sklearn.cluster import DBSCAN
import tarfile
from sklearn.neighbors import KernelDensity

import sys
sys.path.insert(0, "..")
from DEFAULTS import BASE_PATH
from configuration import Configuration

DATAPATH = "/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/TheresaProteins"

class SynProtConfiguration(Configuration):
    num_classes: int = 1
    criterion: str = "MSELoss"
    min_annotated_ratio: float = 0.1

class ProteinSegmentationDataset(Dataset):
    def __init__(
        self,
        archive_path: str,
        transform: Optional[Callable] = None,
        n_channels: int = 1,
    ) -> None:
        self.archive_path = archive_path
        self.transform = transform
        self.n_channels = n_channels
        with tarfile.open(self.archive_path, "r") as archive_obj:
            self.members = list(sorted(archive_obj.getmembers(), key=lambda m: m.name))

    def __len__(self) -> int:
        return len(self.members)
    
    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.Tensor]:
        member = self.members[idx]
        with tarfile.open(self.archive_path, "r") as archive_obj:
            buffer = io.BytesIO()
            buffer.write(archive_obj.extractfile(member).read())
            buffer.seek(0)
            data = np.load(buffer, allow_pickle=True)
            data = {key : values for key, values in data.items()}
            img, mask = data["img"], data["segmentation"]
        if self.n_channels == 3:
            img = np.tile(img[np.newaxis], (3, 1, 1))
            img = torch.tensor(img, dtype=torch.float32)
        else:
            img = torch.tensor(img[np.newaxis, ...], dtype=torch.float32)
        mask = torch.tensor(mask[np.newaxis, ...], dtype=torch.float32)
        if self.transform is not None:
            img = self.transform(img)
        return img, mask

    

# class ProteinSegmentationDataset(Dataset):
#     def __init__(
#         self,
#         h5file: str,
#         transform=None,
#         n_channels=1,
#     ) -> None:
#         self.h5file = h5file

#         if transform is None:
#             self.transform = transforms.ToTensor()
#         else:
#             self.transform = transform

#         self.n_channels = n_channels 
#         self.classes = ['synaptic-proteins']
#         with h5py.File(h5file, "r") as hf:
#             self.dataset_size = hf["images"][()].shape[0] 

#     def __len__(self):
#         return self.dataset_size 

#     def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.Tensor]:
#         with h5py.File(self.h5file, "r") as hf:
#             img = hf["images"][idx]
#             mask = hf["masks"][idx]
        
#         if self.n_channels == 3:
#             img = np.tile(img[np.newaxis], (3, 1, 1))
#             img = np.moveaxis(img, 0, -1)
        
#         img = self.transform(img)
#         mask = transforms.ToTensor()(mask)
#         return img, mask

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

class PerforatedProteinSegmentationDataset(Dataset):

    PERFORATED = 2

    def __init__(self, h5file: str, transform=None, n_channels=1, validation=False, data_aug=0.5) -> None:
        self.h5file = h5file

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

        self.n_channels = n_channels
        self.validation = validation
        self.data_aug = data_aug

        self.half_crop_size = 168
        self.image_size = 224

        self.classes = ['perforated']

        with h5py.File(self.h5file, "r") as file:
            self.valid_indices = numpy.argwhere(file["labels"][()].ravel() == self.PERFORATED).ravel()

            metadata = file["crops"].attrs["metadata"]
            metadata = json.loads(metadata)
            self.metadata = [metadata[idx] for idx in self.valid_indices]
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.Tensor]:
        
        valid_idx = self.valid_indices[idx]
        
        # Retrieves metadata
        metadata = self.metadata[idx]

        with h5py.File(self.h5file, "r") as file:
            image = file["original_images"][metadata["img_name"]][metadata["channel"]]
            coord = metadata["id_coord"]
            slc = tuple(
                slice(max(0, coord[i] - self.half_crop_size), min(coord[i] + self.half_crop_size, image.shape[i]))
                for i in range(len(coord))
            )
            img = image[slc]
            mask = file["masks"][valid_idx][()]

            top_padding = self.half_crop_size - mask.shape[-2] // 2 if coord[0] > self.half_crop_size else coord[0] - mask.shape[-2] // 2
            bottom_padding = self.half_crop_size - mask.shape[-2] // 2 if coord[0] + self.half_crop_size < image.shape[-2] else image.shape[-2] - coord[0] - mask.shape[-2] // 2
            left_padding = self.half_crop_size - mask.shape[-1] // 2 if coord[1] > self.half_crop_size else coord[1] - mask.shape[-1] // 2
            right_padding = self.half_crop_size - mask.shape[-1] // 2 if coord[1] + self.half_crop_size < image.shape[-1] else image.shape[-1] - coord[1] - mask.shape[-1] // 2
            mask = numpy.pad(mask, ((0, 0), (top_padding, bottom_padding), (left_padding, right_padding)), mode="constant", constant_values=0)

        # Normalize image 
        img = (img - img.min()) / (img.max() - img.min())
        img = img.astype(numpy.float32)

        # Create segmentation mask from all votes
        votes = numpy.array(metadata["seg_votes"])
        votes = votes[votes < len(mask)]
        masks = mask[votes]
        mask = numpy.mean(masks, axis=0)
        mask = mask.astype(numpy.float32)

        # Random 224 crop within the image
        if img.shape[-2] < self.image_size:
            img = numpy.pad(img, ((0, self.image_size - img.shape[-2]), (0, 0)), mode="symmetric")
            mask = numpy.pad(mask, ((0, self.image_size - mask.shape[-2]), (0, 0)), mode="symmetric")
        if img.shape[-1] < self.image_size:
            img = numpy.pad(img, ((0, 0), (0, self.image_size - img.shape[-1])), mode="symmetric")
            mask = numpy.pad(mask, ((0, 0), (0, self.image_size - mask.shape[-1])), mode="symmetric")

        coord_y, coord_x = numpy.random.randint(0, [img.shape[-2] - self.image_size + 1, img.shape[-1] - self.image_size + 1])

        image_crop = img[coord_y:coord_y + self.image_size, coord_x:coord_x + self.image_size]
        label_crop = mask[coord_y:coord_y + self.image_size, coord_x:coord_x + self.image_size]
        label_crop = label_crop[numpy.newaxis]
        
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

def combine_coords(pos, x_coords, y_coords):
    full_x, full_y = [], []
    for p in pos:
        all_x, all_y = y_coords[p], x_coords[p]
        if all_x is None or all_y is None:
            continue
        
        #for i, (x, y) in enumerate(zip(all_x, all_y)):
        #    if type(x) == list:
        #        all_x[i] = np.mean(x)
        #        all_y[i] = np.mean(y)
        
        full_x += all_x
        full_y += all_y
            
    # split points vs lines
    
    line_x, line_y = [], []
    point_x, point_y = [], []
    for i in range(len(full_x)):
        if type(full_x[i]) == list:
            line_x.append(full_x[i])
            line_y.append(full_y[i])
        else:
            point_x.append(full_x[i])
            point_y.append(full_y[i])

    point_x, point_y = np.array(point_x)/3, np.array(point_y)/3
    line_x, line_y = np.array(line_x)/3, np.array(line_y)/3
    
    return point_x, point_y, line_x, line_y

def gaussian_map(X, Y):
    canvas = np.zeros((80,80))
    for y, x in zip(Y, X):
        y, x = int(y/3), int(x/3)
        try:
            canvas[int(x), int(y)] += 1
        except IndexError:
            continue
        
        canvas /= np.max(canvas)
        canvas = gaussian_filter(canvas, sigma=2)
    
    return canvas

def kernel_density_map(X, Y, img, bw=2):
    
    data = np.array([X, Y]).swapaxes(0,1)
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(data)
    
    # grid
    x, y = np.arange(0, img.shape[0]), np.arange(0, img.shape[1])
    xx, yy = np.meshgrid(x, y)
    grid_data = np.array([yy.ravel(), xx.ravel()]).swapaxes(0,1)
    
    out = np.exp(kde.score_samples(grid_data))
    
    out = np.reshape(out, img.shape)
    out = out/np.max(out)
    return out

def dbscan_map(X, Y, eps=3.5, min_samples=3):
    data = np.array([X, Y]).swapaxes(0,1)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    while len(np.unique(labels[labels > -1])) < 2:
        eps -= 0.5
        if eps == 0:
            break
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data)
        
    return labels

def find_local_max(kde_map):
    xy = peak_local_max(kde_map, min_distance=2, threshold_abs=0.33)
    
    return xy

def find_cluster_centers(X, Y, labels):
    pass

class MultidomainDetectionDataset(Dataset):

    MULTIDOMAIN = 3

    def __init__(self, h5file: str, transform=None, n_channels=1, validation=False, data_aug=0.5) -> None:
        self.h5file = h5file
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.n_channels = n_channels
        self.validation = validation
        self.data_aug = data_aug

        self.half_crop_size = 168
        self.image_size = 224

        self.classes = ['multidomain']

        with h5py.File(self.h5file, "r") as file:
            valid_indices = numpy.argwhere(file["labels"][()].ravel() == self.MULTIDOMAIN).ravel()

            metadata = file["crops"].attrs["metadata"]
            metadata = json.loads(metadata)
            for i in range(len(metadata)):
                metadata[i]["idx"] = i
            self.metadata = [metadata[idx] for idx in valid_indices]

            self.images = {
                key : {"image" : value[()], "label" : numpy.zeros_like(value[()], dtype=numpy.float32)} for key, value in file["original_images"].items()
            }            
        
        self.create_label_map()
        self.valid_indices = self.get_valid_indices()
    
    def create_label_map(self):
        
        for i, metadata in enumerate(tqdm(self.metadata)):

            label = metadata["class_majority"]
            pos = numpy.where(numpy.array(metadata["class_votes"]) == label)[0]

            x_coords = metadata["x_coord"].replace("nan", "None")
            x_coords = ast.literal_eval(x_coords)
            y_coords = metadata["y_coord"].replace("nan", "None")
            y_coords = ast.literal_eval(y_coords)

            if len(metadata["class_votes"]) != len(x_coords):
                continue

            point_x, point_y, line_x, line_y = combine_coords(pos, x_coords, y_coords)
            if len(point_x) == 0:
                continue
            
            cluster_labels = dbscan_map(point_x, point_y)
            filt_x, filt_y = point_x[cluster_labels != -1], point_y[cluster_labels != -1]
            if len(filt_x) == 0:
                continue
            canvas = kernel_density_map(filt_x, filt_y, np.zeros((80,80)), bw=2)
            yx = find_local_max(canvas)

            # This is center of annotation in large image
            coord = numpy.array(metadata["id_coord"])
            
            idx = metadata["idx"]
            with h5py.File(self.h5file, "r") as file:
                crop = file["crops"][idx][()]
            shift, error, diffphase = phase_cross_correlation(
                crop, canvas, upsample_factor=1, normalization=None)
            coord = (coord + shift).astype(int)

            # Skip regions that are out of bounds
            if coord[0] - canvas.shape[-2] // 2 < 0 or coord[1] - canvas.shape[-1] // 2 < 0:
                continue
            if coord[0] + canvas.shape[-2] // 2 > self.images[metadata["img_name"]]["label"].shape[-2] or coord[1] + canvas.shape[-2] // 2 > self.images[metadata["img_name"]]["label"].shape[-1]:
                continue

            self.images[metadata["img_name"]]["label"][metadata["channel"]][
                coord[0] - canvas.shape[-2] // 2 : coord[0] + canvas.shape[-2] // 2,
                coord[1] - canvas.shape[-1] // 2 : coord[1] + canvas.shape[-1] // 2] += canvas
                    
    def get_valid_indices(self):
        valid_indices = []
        for key, values in self.images.items():
            img = values["image"]
            label = values["label"]
            for chan in range(img.shape[0]):
                for j in range(0, img.shape[-2], self.image_size):
                    for i in range(0, img.shape[-1], self.image_size):
                        if numpy.sum(label[chan, j:j+self.image_size, i:i+self.image_size]) > 0:
                            valid_indices.append((key, chan, j, i))
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.Tensor]:
        
        key, chan, j, i = self.valid_indices[idx]
        
        image_crop = self.images[key]["image"][chan, j:j+self.image_size, i:i+self.image_size]
        label_crop = self.images[key]["label"][chan, j:j+self.image_size, i:i+self.image_size]
        label_crop = label_crop[numpy.newaxis]

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

def get_dataset(name, cfg, **kwargs):
    cfg.dataset_cfg = SynProtConfiguration()

    if cfg.in_channels == 3:
        # ImageNet normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.0695771782959453, 0.0695771782959453, 0.0695771782959453], std=[0.12546228631005282, 0.12546228631005282, 0.12546228631005282])
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.ToTensor()

    if name == "synaptic-protein-segmentation":
        datapath = os.path.join(BASE_PATH, "Datasets/Neural-Activity-States/PSD95-Basson")
        train_dataset = ProteinSegmentationDataset(
            archive_path=f"{datapath}/synaptic-protein-segmentation_train.tar",
            transform=None, 
            n_channels=cfg.in_channels
        )
        valid_dataset = ProteinSegmentationDataset(
            archive_path=f"{datapath}/synaptic-protein-segmentation_valid.tar",
            transform=None, 
            n_channels=cfg.in_channels
        )
        test_dataset = ProteinSegmentationDataset(
            archive_path=f"{datapath}/synaptic-protein-segmentation_test.tar",
            transform=None, 
            n_channels=cfg.in_channels
        )
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Valid dataset size: {len(valid_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        return train_dataset, valid_dataset, test_dataset
    
    if name == "synaptic-semantic-segmentation":

        train_dataset = SemanticProteinSegmentationDataset(
            h5file=os.path.join(BASE_PATH, "segmentation-data", "synprot", "train_2024-05-16.hdf5"),
            transform=transform, 
            n_channels=cfg.in_channels
        )
        valid_dataset = SemanticProteinSegmentationDataset(
            h5file=os.path.join(BASE_PATH, "segmentation-data", "synprot", "valid_2024-05-16.hdf5"),
            transform=transform, 
            n_channels=cfg.in_channels
        )
        test_dataset = SemanticProteinSegmentationDataset(
            h5file=os.path.join(BASE_PATH, "segmentation-data", "synprot", "test_2024-05-16.hdf5"),
            transform=transform, 
            n_channels=cfg.in_channels
        )

        # Updates num classes
        cfg.dataset_cfg.num_classes = len(train_dataset.classes)

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Valid dataset size: {len(valid_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        return train_dataset, valid_dataset, test_dataset

    elif name == "perforated-segmentation":
        train_dataset = PerforatedProteinSegmentationDataset(
            h5file=os.path.join(BASE_PATH, "segmentation-data", "synprot", "train_2024-05-16.hdf5"),
            transform=transform, 
            n_channels=cfg.in_channels
        )
        valid_dataset = PerforatedProteinSegmentationDataset(
            h5file=os.path.join(BASE_PATH, "segmentation-data", "synprot", "valid_2024-05-16.hdf5"),
            transform=transform, 
            n_channels=cfg.in_channels,
            validation=True
        )
        test_dataset = PerforatedProteinSegmentationDataset(
            h5file=os.path.join(BASE_PATH, "segmentation-data", "synprot", "test_2024-05-16.hdf5"),
            transform=transform, 
            n_channels=cfg.in_channels,
            validation=True
        )
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Valid dataset size: {len(valid_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        return train_dataset, valid_dataset, test_dataset

    elif name == "multidomain-detection":
        train_dataset = MultidomainDetectionDataset(
            h5file=os.path.join(BASE_PATH, "segmentation-data", "synprot", "train_2024-05-16.hdf5"),
            transform=transform, 
            n_channels=cfg.in_channels
        )
        valid_dataset = MultidomainDetectionDataset(
            h5file=os.path.join(BASE_PATH, "segmentation-data", "synprot", "valid_2024-05-16.hdf5"),
            transform=transform, 
            n_channels=cfg.in_channels,
            validation=True
        )
        test_dataset = MultidomainDetectionDataset(
            h5file=os.path.join(BASE_PATH, "segmentation-data", "synprot", "test_2024-05-16.hdf5"),
            transform=transform, 
            n_channels=cfg.in_channels,
            validation=True
        )
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Valid dataset size: {len(valid_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        return train_dataset, valid_dataset, test_dataset