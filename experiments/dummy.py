"""
Script mostly used for displaying quick info or downloading torchvision models on login nodes (w/ internet)
"""
import torch
import numpy as np
from model_builder import get_pretrained_model, get_pretrained_model_v2, get_base_model
from timm.models.vision_transformer import vit_small_patch16_224, vit_base_patch16_224, vit_tiny_patch16_224, vit_large_patch16_224
import argparse 
import torchvision
from torchinfo import summary
from loaders import get_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List
import h5py
import glob
import random

PATH = "/home/frbea320/scratch/Datasets/SynapticProteins/dataset"
OUTPATH = "/home/frbea320/scratch/Datasets/FLCDataset/TheresaProteins"
THRESHOLD = 0.001
protein_dict = {
    "Bassoon": 0,
    "Homer": 1,
    "NKCC2": 2,
    "Rim": 3,
    "PSD95": 4,
}

condition_dict = {
    "0MgGlyBic": 0,
    "Block": 1,
    "GluGly": 2,
    "KCl": 3,
    "4hTTX": 4,
    "24hTTX": 5,
    "48hTTX": 6,
    "naive": 7,
}

def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def get_files(path: str = "/home/frbea320/scratch/Datasets/SynapticProteins/dataset"):
    files = glob.glob(f"{path}/**/*.npz", recursive=True)
    files = list(set(files))
    print(len(files))
    return files

def create_hdf5(files: List[str], path: str = OUTPATH):
    with h5py.File(f"{path}/train_segmentation.hdf5", "a") as train_hf:
        with h5py.File(f"{path}/valid_segmentation.hdf5", "a") as valid_hf:
            with h5py.File(f"{path}/test_segmentation.hdf5", "a") as test_hf:
                # Initialize h5 Datasets
                train_imgs = train_hf.create_dataset(name="images", shape=(0, 224, 224), maxshape=(None, 224, 224))
                valid_imgs = valid_hf.create_dataset(name="images", shape=(0, 224, 224), maxshape=(None, 224, 224))
                test_imgs = test_hf.create_dataset(name="images", shape=(0, 224, 224), maxshape=(None, 224, 224))
                train_masks = train_hf.create_dataset(name="masks", shape=(0, 224, 224), maxshape=(None, 224, 224))
                valid_masks = valid_hf.create_dataset(name="masks", shape=(0, 224, 224), maxshape=(None, 224, 224))
                test_masks = test_hf.create_dataset(name="masks", shape=(0, 224, 224), maxshape=(None, 224, 224))

                # Fill Datasets
                pbar = tqdm(files, total=len(files))
                background_crops = 0
                train_crops = 0
                valid_crops = 0
                test_crops = 0
                for f in pbar:
                    data = np.load(f)
                    imgs = data['img'][0], data['img'][1]
                    masks = data['mask'][0], data['mask'][1]
                    for img, mask in zip(imgs, masks):
                        img = normalize(img)
                        num_y = np.floor(img.shape[0] / 224)
                        num_x = np.floor(img.shape[1] / 224)
                        ys = np.arange(0, num_y*224, 224).astype('int')
                        xs = np.arange(0, num_x*224, 224).astype('int')
                        for y in ys:
                            for x in xs:
                                crop = img[y:y+224, x:x+224]
                                mask_crop = mask[y:y+224, x:x+224]
                                assert crop.shape == mask_crop.shape
                                pixels = crop.shape[0] * crop.shape[1]
                                foreground = np.count_nonzero(mask_crop)
                                ratio = foreground / pixels
                                if ratio <= THRESHOLD:
                                    background_crops += 1
                                    continue
                                else:
                                    prob = random.random()
                                    if prob < 0.80:
                                        train_imgs.resize(train_imgs.shape[0] + 1, axis=0)
                                        train_masks.resize(train_masks.shape[0] + 1, axis=0)
                                        train_imgs[-1:] = crop
                                        train_masks[-1:] = mask_crop
                                        train_crops += 1
                                    elif prob >= 0.80 and prob < 0.90:
                                        valid_imgs.resize(valid_imgs.shape[0] + 1, axis=0)
                                        valid_masks.resize(valid_masks.shape[0] + 1, axis=0)
                                        valid_imgs[-1:] = crop
                                        valid_masks[-1:] = mask_crop
                                        valid_crops += 1
                                    else: # prob >= 0.90
                                        test_imgs.resize(test_imgs.shape[0] + 1, axis=0)
                                        test_masks.resize(test_masks.shape[0] + 1, axis=0)
                                        test_imgs[-1:] = crop
                                        test_masks[-1:] = mask_crop
                                        test_crops += 1
                    pbar.set_description(f"Train crops: {train_crops}\nValid crops: {valid_crops}\nTest crops: {test_crops}\nBackground crops: {background_crops}")


def check_hdf5():
    indices = np.sort(np.random.randint(0, 70000, size=20))
    with h5py.File(f"{OUTPATH}/train_segmentation.hdf5", "r") as hf:
        images = hf["images"][indices]
        masks = hf["masks"][indices]

    counter = 0
    for img, mask in zip(images, masks):
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img, cmap='hot')
        axs[1].imshow(mask, cmap='gray')
        plt.tight_layout()
        fig.savefig(f"./temp{counter}.png", dpi=1200)
        plt.close(fig)
        counter += 1

def main():
    check_hdf5()


if __name__=="__main__":
    main()