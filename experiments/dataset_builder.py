import torch 
import numpy as np 
from tqdm import tqdm 
from typing import List, Tuple, Union 
import tarfile 
import glob 
import io
import matplotlib.pyplot as plt 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="/home/frederic/Datasets/WaveletProteins")
parser.add_argument("--outpath", type=str, default="/home-local/Frederic/evaluation-data/NeuralActivityStates")
args = parser.parse_args()

def load_filenames(path: str, train_percentage: float = 0.70, valid_percentage: float = 0.15) -> List[str]:
    files = glob.glob(f"{path}/PSD95-Bassoon/*.npz")
    files = list(set(files))
    valid_conditions = ["Block", "0MgGlyBic", "GluGly", "48hTTX"]
    conditions = [item.split("/")[-1].split(".")[0].split("_")[-2] for item in files]
    filtered_files = []
    filtered_conditions = []
    for f, c in zip(files, conditions):
        if c in valid_conditions:
            filtered_files.append(f)
            filtered_conditions.append(c)
    uniques, counts = np.unique(filtered_conditions, return_counts=True)
    all_train_files, all_valid_files, all_test_files = [], [], [] 
    train_conditions, valid_conditions, test_conditions = [], [], []
    for u, c in zip(uniques, counts):
        cond_files = [f for f, c in zip(filtered_files, filtered_conditions) if c == u] 
        N = len(cond_files)
        train_files = np.random.choice(cond_files, size=int(train_percentage * N), replace=False)
        cond_files = np.setdiff1d(cond_files, train_files) 
        valid_files = np.random.choice(cond_files, size=int(valid_percentage * N), replace=False)
        test_files = np.setdiff1d(cond_files, valid_files)
        all_train_files.extend(train_files)
        all_valid_files.extend(valid_files)
        all_test_files.extend(test_files)
        train_conditions.extend([u] * len(train_files))
        valid_conditions.extend([u] * len(valid_files))
        test_conditions.extend([u] * len(test_files))

    print("=== Train ===")
    print(np.unique(train_conditions, return_counts=True))
    print("=== Valid ===")
    print(np.unique(valid_conditions, return_counts=True))
    print("=== Test ===")
    print(np.unique(test_conditions, return_counts=True))
    print("===========")
    return all_train_files, all_valid_files, all_test_files

def main():
    train_files, valid_files, test_files = load_filenames(path=args.path)
    PROTEIN = "PSD95"
    CROP_SIZE = 224
    THRESHOLD = 0.005 # After qualitative examination of the crops
    total_train_crops = 0
    total_valid_crops = 0
    total_test_crops = 0
    counter = 0
    with tarfile.open(f"{args.outpath}/NAS_train_v2.tar", "a") as handle:
        all_names = []
        for i, f in enumerate(tqdm(train_files, desc="...Train files...")):
            data = np.load(f)
            img, mask = data["img"][0], data["mask"][0] # Indexing 0 because we will always be taking the PSD95 image
            m, M = np.quantile(img, 0.0001), np.quantile(img, 0.9999)
            img = (img - m) / (M - m)
            img = np.clip(img, 0, 1)
            img = img.astype(np.float32)
            condition = f.split("/")[-1].split(".")[0].split("_")[-2]
            num_y = np.floor(img.shape[0] / CROP_SIZE)
            num_x = np.floor(img.shape[1] / CROP_SIZE)
            ys = np.arange(0, num_y*CROP_SIZE, CROP_SIZE).astype(np.int64)
            xs = np.arange(0, num_x*CROP_SIZE, CROP_SIZE).astype(np.int64)

            for y in ys:
                for x in xs:
                    crop = img[y:y+CROP_SIZE, x:x+CROP_SIZE]
                    mask_crop = mask[y:y+CROP_SIZE, x:x+CROP_SIZE]
                    foreground = np.count_nonzero(mask_crop)
                    pixels = crop.shape[0] * crop.shape[1]
                    ratio = foreground / pixels
                    if ratio < THRESHOLD:
                        continue 
                    else:
                        buffer = io.BytesIO()
                        np.savez(buffer, image=crop, mask=mask_crop, metadata={"condition": condition, "protein": PROTEIN, "img_folder": "PSD95-Bassoon"})
                        buffer.seek(0)
                        name = f"{condition}-{PROTEIN}-{counter}"
                        assert name not in all_names
                        all_names.append(name)
                        tarinfo = tarfile.TarInfo(name=name)
                        tarinfo.size = len(buffer.getbuffer())
                        handle.addfile(tarinfo=tarinfo, fileobj=buffer)
                        counter += 1
                        total_train_crops += 1
    print(len(all_names))

    with tarfile.open(f"{args.outpath}/NAS_valid_v2.tar", "a") as handle:
        counter = 0
        all_names = []
        for i, f in enumerate(tqdm(valid_files, desc="...Valid files...")):
            data = np.load(f)
            img, mask = data["img"][0], data["mask"][0] # Indexing 0 because we will always be taking the PSD95 image
            m, M = np.quantile(img, 0.0001), np.quantile(img, 0.9999)
            img = (img - m) / (M - m)
            img = np.clip(img, 0, 1)
            img = img.astype(np.float32)
            condition = f.split("/")[-1].split(".")[0].split("_")[-2]
            num_y = np.floor(img.shape[0] / CROP_SIZE)
            num_x = np.floor(img.shape[1] / CROP_SIZE)
            ys = np.arange(0, num_y*CROP_SIZE, CROP_SIZE).astype(np.int64)
            xs = np.arange(0, num_x*CROP_SIZE, CROP_SIZE).astype(np.int64)
            for y in ys:
                for x in xs:
                    crop = img[y:y+CROP_SIZE, x:x+CROP_SIZE]
                    mask_crop = mask[y:y+CROP_SIZE, x:x+CROP_SIZE]
                    foreground = np.count_nonzero(mask_crop)
                    pixels = crop.shape[0] * crop.shape[1]
                    ratio = foreground / pixels
                    if ratio < THRESHOLD:
                        continue 
                    else:
                        buffer = io.BytesIO()
                        np.savez(buffer, image=crop, mask=mask_crop, metadata={"condition": condition, "protein": PROTEIN, "img_folder": "PSD95-Bassoon"})
                        buffer.seek(0)
                        name = f"{condition}-{PROTEIN}-{counter}"
                        assert name not in all_names
                        all_names.append(name)
                        tarinfo = tarfile.TarInfo(name=name)
                        tarinfo.size = len(buffer.getbuffer())
                        handle.addfile(tarinfo=tarinfo, fileobj=buffer)
                        counter += 1
                        total_valid_crops += 1
    print(len(all_names))


    with tarfile.open(f"{args.outpath}/NAS_test_v2.tar", "a") as handle:
        counter = 0
        all_names = []
        for i, f in enumerate(tqdm(test_files, desc="...Test files...")):
            data = np.load(f)
            img, mask = data["img"][0], data["mask"][0] # Indexing 0 because we will always be taking the PSD95 image
            m, M = np.quantile(img, 0.0001), np.quantile(img, 0.9999)
            img = (img - m) / (M - m)
            img = np.clip(img, 0, 1)
            img = img.astype(np.float32)
            condition = f.split("/")[-1].split(".")[0].split("_")[-2]
            num_y = np.floor(img.shape[0] / CROP_SIZE)
            num_x = np.floor(img.shape[1] / CROP_SIZE)
            ys = np.arange(0, num_y*CROP_SIZE, CROP_SIZE).astype(np.int64)
            xs = np.arange(0, num_x*CROP_SIZE, CROP_SIZE).astype(np.int64)
            for y in ys:
                for x in xs:
                    crop = img[y:y+CROP_SIZE, x:x+CROP_SIZE]
                    mask_crop = mask[y:y+CROP_SIZE, x:x+CROP_SIZE]
                    foreground = np.count_nonzero(mask_crop)
                    pixels = crop.shape[0] * crop.shape[1]
                    ratio = foreground / pixels
                    if ratio < THRESHOLD:
                        continue 
                    else:
                        buffer = io.BytesIO()
                        np.savez(buffer, image=crop, mask=mask_crop, metadata={"condition": condition, "protein": PROTEIN, "img_folder": "PSD95-Bassoon"})
                        buffer.seek(0)
                        name = f"{condition}-{PROTEIN}-{counter}"
                        assert name not in all_names
                        all_names.append(name)  
                        tarinfo = tarfile.TarInfo(name=name)
                        tarinfo.size = len(buffer.getbuffer())
                        handle.addfile(tarinfo=tarinfo, fileobj=buffer)
                        counter += 1
                        total_test_crops += 1
    print(len(all_names))


    print("=== Summary ===")
    print(f"\tTotal train crops: {total_train_crops}")
    print(f"\tTotal valid crops: {total_valid_crops}")
    print(f"\tTotal test crops: {total_test_crops}")
    print("===============")
if __name__=="__main__":
    main()
