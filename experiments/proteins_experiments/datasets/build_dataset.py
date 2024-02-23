import numpy as np 
import h5py 
import pickle
import glob
from tqdm import tqdm 
from typing import List
import random 
from skimage.filters import threshold_otsu

THRESHOLD = 0.001

INPATH = "/home/frbea320/scratch/Datasets/AnomalyDetectionDatasets/randomly_located_synthetic_anomalies/normal"
OUTPATH = "/home/frbea320/scratch/Datasets/FLCDataset/TheresaProteins"

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

def get_image_cornichons(path: str = INPATH) -> List[str]:
    files = glob.glob(f"{path}/*.pkl")
    files = list(set(files))
    return files 


def create_hdf5(files: List[str], path: str = OUTPATH) -> None:
    with h5py.File(f"{path}/theresa_proteins.hdf5", "a") as hf:
        img_set = hf.create_dataset(name="images", shape=(0, 224, 224), maxshape=(None, 224, 224))
        protein_set = hf.create_dataset(name="protein", shape=(0, ), maxshape=(None, ))
        condition_set = hf.create_dataset(name="condition", shape=(0, ), maxshape=(None, ))
        pbar = tqdm(files, total=len(files))
        background_crops = 0
        for f in pbar:
            with open(f, "rb") as handle:
                image_dict = pickle.load(handle)
            img = image_dict['img']
            tau = threshold_otsu(img)
            img_tau = img >= tau
            protein = protein_dict[image_dict['protein']]
            condition = condition_dict[image_dict['condition']]
            num_y = np.floor(img.shape[0] / 224)
            num_x = np.floor(img.shape[1] / 224)
            ys = np.arange(0, num_y * 224, 224).astype('int')
            xs = np.arange(0, num_x * 224, 224).astype('int')
            for y in ys:
                for x in xs:
                    crop = img[y:y+224, x:x+224]
                    crop_tau = img_tau[y:y+224, x:x+224]
                    pixels = crop.shape[0] * crop.shape[1]
                    foreground = np.count_nonzero(crop_tau)
                    ratio = foreground / pixels
                    if ratio < THRESHOLD:
                        background_crops += 1
                        continue
                    img_set.resize(img_set.shape[0] + 1, axis=0)
                    protein_set.resize(protein_set.shape[0] + 1, axis=0)
                    condition_set.resize(condition_set.shape[0] + 1, axis=0)
                    img_set[-1:] = crop
                    protein_set[-1:] = protein
                    condition_set[-1:] = condition
                    pbar.set_description(f"Ignored {background_crops} background crops so far")   
    print(f"Ignored a total of {background_crops} background_crops")
    print("********* DONE *************")           

def main():
    files = get_image_cornichons()
    create_hdf5(files=files)

if __name__=="__main__":
    main()
