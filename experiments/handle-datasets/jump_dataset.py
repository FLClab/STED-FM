import os 
import tarfile 
import numpy as np 
import io 
import tifffile 
import argparse 
from tqdm import tqdm 
import glob
import os
from skimage import filters 

parser = argparse.ArgumentParser() 
parser.add_argument("--overwrite", action="store_true", help="Overwrites the tar file")
args = parser.parse_args()

BASEPATH = "/home/frbea320/scratch/Datasets/JUMP_CP/cpg0016-jump"
OUTPATH = "/home/frbea320/scratch/Datasets/JUMP_CP"

MINIMUM_FOREGROUND = 0.01 
CROP_SIZE = 224 

def load_filenames():
    all_files = []
    for root, dirs, files in os.walk(BASEPATH):

        for file in files:
            if file.endswith(".tif") or file.endswith(".tiff"):
                all_files.append(os.path.join(root, file))
    return list(set(all_files)) # ensuring no duplicates



def main():
    files = load_filenames()

    with tarfile.open(f"{OUTPATH}/jump.tar", "a") as handle:
        for f in files:
            image = tifffile.imread(f)
            if (image.shape[-2] < CROP_SIZE) or (image.shape[-1] < CROP_SIZE) or (image.ndim !=2):
                continue

            m, M = np.min(image), np.max(image)
            image = (image - m) / (M - m)
            image = (image*255).astype(np.uint8)

            buffer = io.BytesIO()
            np.savez(buffer, image=image)
            buffer.seek(0)

            tarinfo = tarfile.TarInfo(name=f"{f}")
            tarinfo.size = len(buffer.getbuffer())
            handle.addfile(tarinfo=tarinfo, fileobj=buffer)

if __name__=="__main__":
    main()