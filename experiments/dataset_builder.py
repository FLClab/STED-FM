import torch 
import numpy as np
from tqdm import tqdm 
from typing import List, Tuple
import h5py 
import glob 
import random 
import tifffile
from skimage.filters import threshold_otsu

def normalize(img):
    return (img - np.quantile(img, 0.001)) / (np.quantile(img, 0.999) - np.quantile(img, 0.001))

PATH = "/home/frbea320/scratch/Datasets/SynapticProteins/dataset"
OUTPATH = "/home/frbea320/scratch/Datasets/FLCDataset/TheresaProteins"
THRESHOLD = 0.001

protein_dict = {
    "Bassoon": 0,
    "Homer": 1,
    "Rim": 2,
    "PSD95": 3,
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

def get_tiff_files():
    folders = ["Bassoon-Homer", "Bassoon-Rim", "PSD95-Bassoon"]
    files = glob.glob(f"/home/frbea320/scratch/Datasets/SynapticProteins/**/**/*.tif")
    files = list(set(files))
    files = [item for item in files if item.split("/")[-3] in folders]
    return files

def create_hdf5(files: List[str], path: str = OUTPATH, crop_size: int = 224) -> None:
    with h5py.File(f"{path}/train_v2.hdf5", "a") as train_handle:
        with h5py.File(f"{path}/valid_v2.hdf5", "a") as valid_handle:
            with h5py.File(f"{path}/test_v2.hdf5", "a") as test_handle:
                train_imgs = train_handle.create_dataset(name="images", shape=(0, 224, 224), maxshape=(None, 224, 224))
                valid_imgs = valid_handle.create_dataset(name="images", shape=(0, 224, 224), maxshape=(None, 224, 224))
                test_imgs = test_handle.create_dataset(name="images", shape=(0, 224, 224), maxshape=(None, 224, 224))

                train_proteins = train_handle.create_dataset(name="proteins", shape=(0,), maxshape=(None, ))
                valid_proteins = valid_handle.create_dataset(name="proteins", shape=(0,), maxshape=(None, ))
                test_proteins = test_handle.create_dataset(name="proteins", shape=(0,), maxshape=(None, ))

                train_conditions = train_handle.create_dataset(name="conditions", shape=(0,), maxshape=(None,))
                valid_conditions = valid_handle.create_dataset(name="conditions", shape=(0,), maxshape=(None,))
                test_conditions = test_handle.create_dataset(name="conditions", shape=(0,), maxshape=(None,))

                for f in tqdm(files, desc="Files..."):
                    fsplit = f.split("/")
                    proteins, condition, fname = fsplit[-3:]
                    proteinsplit = proteins.split("-")
                    fname = fname.replace(".tif", "")
                    data = tifffile.imread(f)

                    channels = [0, 1] if proteins == "PSD95-Bassoon" else [1] 
                    for ch in channels:
                        p = protein_dict[proteinsplit[ch]]
                        # print(p)
                        c = condition_dict[condition]
                        img = normalize(data[ch])
                        tau = threshold_otsu(img)
                        img_tau = img >= tau 
                        num_y = np.floor(img.shape[0] / crop_size)
                        num_x = np.floor(img.shape[1] / crop_size)
                        ys = np.arange(0, num_y*crop_size, crop_size).astype(np.int64)
                        xs = np.arange(0, num_x*crop_size, crop_size).astype(np.int64)
                        for y in ys:
                            for x in xs:
                                crop = img[y:y+crop_size, x:x+crop_size]
                                tau_crop = img_tau[y:y+crop_size, x:x+crop_size]
                                assert crop.shape == tau_crop.shape
                                pixels = crop.shape[0] * crop.shape[1]
                                foreground = np.count_nonzero(tau_crop)
                                ratio = foreground / pixels
                                if ratio < THRESHOLD:
                                    continue

                                set_prob = random.random()
                                if set_prob <= 0.80:
                                    train_imgs.resize(train_imgs.shape[0] + 1, axis=0)
                                    train_proteins.resize(train_proteins.shape[0] + 1, axis=0)
                                    train_conditions.resize(train_conditions.shape[0] + 1, axis=0)
                                    train_imgs[-1:] = crop
                                    train_proteins[-1:] = p 
                                    train_conditions[-1:] = c
                                elif set_prob > 0.80 and set_prob <= 0.90:
                                    valid_imgs.resize(valid_imgs.shape[0] + 1, axis=0)
                                    valid_proteins.resize(valid_proteins.shape[0] + 1, axis=0)
                                    valid_conditions.resize(valid_conditions.shape[0] + 1, axis=0)
                                    valid_imgs[-1:] = crop
                                    valid_proteins[-1:] = p 
                                    valid_conditions[-1:] = c
                                else:
                                    test_imgs.resize(test_imgs.shape[0] + 1, axis=0)
                                    test_proteins.resize(test_proteins.shape[0] + 1, axis=0)
                                    test_conditions.resize(test_conditions.shape[0] + 1, axis=0)
                                    test_imgs[-1:] = crop
                                    test_proteins[-1:] = p
                                    test_conditions[-1:] = c

  
def main():
    files = get_tiff_files()
    create_hdf5(files=files)

if __name__=="__main__":
    main()