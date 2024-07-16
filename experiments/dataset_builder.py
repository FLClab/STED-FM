import torch 
import numpy as np
from tqdm import tqdm 
from typing import List, Tuple, Union
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

def check_conditions(array: Union[List, np.ndarray], target: List = ["0MgGlyBic", "GluGly", "Block", "48hTTX"]):
    return set(target) <= set(array)

def get_tiff_files():
    folders = ["Bassoon-Homer", "Bassoon-Rim", "PSD95-Bassoon"]
    files = glob.glob(f"/home/frbea320/scratch/Datasets/SynapticProteins/**/**/*.tif")
    files = list(set(files))
    files = [item for item in files if item.split("/")[-3] in folders]
    N = len(files)
    val_N = int(0.10 * N)
    test_N = int(0.10 * N)
    test_files = np.random.choice(files, size=val_N)
    train_files = [item for item in files if item not in test_files]
    val_files = np.random.choice(train_files, size=test_N)
    train_files = [item for item in train_files if item not in val_files]
    return train_files, val_files, test_files

def create_hdf5(train_files: List[str],
                valid_files: List[str],
                test_files: List[str],
                path: str = OUTPATH, 
                crop_size: int = 224
                ) -> None:
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


                pbar = tqdm(train_files, desc="Train files...")
                for f in pbar:
                    fsplit = f.split("/")
                    proteins, condition, fname = fsplit[-3:]
                    pbar.set_description(condition)
                    proteinsplit = proteins.split("-")
                    fname = fname.replace(".tif", "")
                    data = tifffile.imread(f)

                    channels = [0, 1] if proteins == "PSD95-Bassoon" else [1] 
                    # set_prob = random.random()
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

                                train_imgs.resize(train_imgs.shape[0] + 1, axis=0)
                                train_proteins.resize(train_proteins.shape[0] + 1, axis=0)
                                train_conditions.resize(train_conditions.shape[0] + 1, axis=0)
                                train_imgs[-1:] = crop
                                train_proteins[-1:] = p 
                                train_conditions[-1:] = c
                print(np.unique(train_handle["conditions"][()]))
                print(np.unique(valid_handle["conditions"][()]))
                print(np.unique(test_handle["conditions"][()]))
                print("------\n")
                    
                   
                               
                pbar = tqdm(valid_files, desc="Validation files...")
                for f in pbar:
                    fsplit = f.split("/")
                    proteins, condition, fname = fsplit[-3:]
                    pbar.set_description(condition)
                    proteinsplit = proteins.split("-")
                    fname = fname.replace(".tif", "")
                    data = tifffile.imread(f)

                    channels = [0, 1] if proteins == "PSD95-Bassoon" else [1] 
                    # set_prob = random.random()
                    for ch in channels:
                        p = protein_dict[proteinsplit[ch]]
                        
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

                                valid_imgs.resize(valid_imgs.shape[0] + 1, axis=0)
                                valid_proteins.resize(valid_proteins.shape[0] + 1, axis=0)
                                valid_conditions.resize(valid_conditions.shape[0] + 1, axis=0)
                                valid_imgs[-1:] = crop
                                valid_proteins[-1:] = p 
                                valid_conditions[-1:] = c
                print("\n------")
                print(np.unique(train_handle["conditions"][()]))
                print(np.unique(valid_handle["conditions"][()]))
                print(np.unique(test_handle["conditions"][()]))
                print("------\n")


                pbar = tqdm(test_files, desc="Test files...")
                for f in pbar:
                    fsplit = f.split("/")
                    proteins, condition, fname = fsplit[-3:]
                    pbar.set_description(condition)
                    proteinsplit = proteins.split("-")
                    fname = fname.replace(".tif", "")
                    data = tifffile.imread(f)

                    channels = [0, 1] if proteins == "PSD95-Bassoon" else [1] 
                    # set_prob = random.random()
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

                                test_imgs.resize(test_imgs.shape[0] + 1, axis=0)
                                test_proteins.resize(test_proteins.shape[0] + 1, axis=0)
                                test_conditions.resize(test_conditions.shape[0] + 1, axis=0)
                                test_imgs[-1:] = crop
                                test_proteins[-1:] = p
                                test_conditions[-1:] = c
                print("\n------")
                print(np.unique(train_handle["conditions"][()]))
                print(np.unique(valid_handle["conditions"][()]))
                print(np.unique(test_handle["conditions"][()]))
                print("------\n")
                    

  
def main():
    good_to_go = False
    while not good_to_go:
        train_files, val_files, test_files = get_tiff_files()
        train_conditions = [item.split("/")[-2] for item in train_files if item.split("/")[-3] == "PSD95-Bassoon"]
        val_conditions = [item.split("/")[-2] for item in val_files if item.split("/")[-3] == "PSD95-Bassoon"]
        test_conditions = [item.split("/")[-2] for item in test_files if item.split("/")[-3] == "PSD95-Bassoon"]
        good_to_go = check_conditions(train_conditions) and check_conditions(val_conditions) and check_conditions(test_conditions)
    print(np.unique(train_conditions))
    print(np.unique(val_conditions))
    print(np.unique(test_conditions))
    print("\n")
    print(check_conditions(train_conditions))
    print(check_conditions(val_conditions))
    print(check_conditions(test_conditions))

    create_hdf5(
        train_files=train_files,
        valid_files=val_files,
        test_files=test_files
        )

if __name__=="__main__":
    main()