import numpy as np
import shutil
import glob
import os
from tqdm import tqdm
ROOT = "/Users/fredbeaupre/valeria-s3/flclab-abberior-sted/jmbellavance"
FOLDERS = [f"{ROOT}/{path}" for path in ["2023-01-12 (25) (cryodendrites)", "2023-02-15 (26.1) (dendrites 4c)", "2023-03-01 (26.2) (dendrites 4c Batch3)"]]
OUTPATH = "/Users/fredbeaupre/valeria-s3/flclab-private/FLCDataset/jmbellavance/ALS_FUS_and_PSD95"

channel_dict = {
    "STED 594 {6}": "FUS",
    "STED 635P {6}": "psd95"
}

def get_files(path: str) -> list:
    files = glob.glob(f"{path}/**/dend*/*.msr")
    return files    

def copy_files(files: list, destination: str, outpath: str = OUTPATH) -> None:
    for f in tqdm(files):
        fname = f.split("/")[-1]
        if os.path.isfile(f):
            shutil.copy(f, f"{outpath}/{destination}/{fname}")
            # print(f"Copied file {f}")

def main():
    for folder_name in FOLDERS:
        dest = folder_name.split("/")[-1]
        if not os.path.exists(f"{OUTPATH}/{dest}"):
            os.mkdir(f"{OUTPATH}/{dest}")
        files = get_files(path=folder_name)
        copy_files(files=files, destination=dest)

if __name__=="__main__":
    main()