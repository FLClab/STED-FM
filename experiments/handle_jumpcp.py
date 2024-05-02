import numpy as np
import glob
import os
import tifffile
import matplotlib.pyplot as plt
from typing import List
import h5py
from tqdm import tqdm

ROOT = "/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/JUMP_CP/cpg0016-jump"


def normalize(img: np.ndarray) -> np.ndarray:
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def get_all_files(path: str = ROOT):
    all_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".tif") or file.endswith(".tiff"):
                all_files.append(os.path.join(root, file))
    return all_files

def create_hdf5(files: List[str]) -> None:
    with h5py.File("/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/JUMP_CP/jump.hdf5", "a") as hf:
        img_set = hf.create_dataset(name="images", shape=(0, 224, 224), maxshape=(None, 224, 224))
        pbar = tqdm(files, total=len(files))
        num_crops = 0
        for f in pbar:
            img = tifffile.imread(f)
            if img.shape[0] < 224 or img.shape[1] < 224:
                continue
            else:
                img = (normalize(img)*255).astype(np.int8)
                num_y = np.floor(img.shape[0] / 224)
                num_x = np.floor(img.shape[1] / 224)
                ys = np.arange(0, num_y * 224, 224).astype(np.int64)
                xs = np.arange(0, num_x * 224, 224).astype(np.int64)
                for y in ys:
                    for x in xs:
                        crop = img[y:y+224, x:x+224]
                        img_set.resize(img_set.shape[0] + 1, axis=0)
                        img_set[-1:] = crop 
                        num_crops += 1
            if num_crops >= 1.3e6:
                break
            pbar.set_description(f"{num_crops} crops.")


def main():
    files = get_all_files()
    create_hdf5(files=files)

    

if __name__=="__main__":
    main()
