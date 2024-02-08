
import os 
import json
import tarfile
import numpy 
import io
import javabridge
import tifffile
import argparse 

from tqdm.auto import tqdm
from skimage import filters

from utils.msrreader import MSRReader

BASEPATH = "/home-local2/projects/FLCDataset"
OUTPATH = "/home-local2/projects/FLCDataset/dataset.tar"
CROP_SIZE = 224
MINIMUM_FOREGROUND = 0.001

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Overwrites the tar file")
    args = parser.parse_args()

    metadata = json.load(open("./datasets/metadata.json", "r"))

    if args.overwrite:
        with tarfile.open(OUTPATH, "w") as tf:
            image_ids = []
    else:
        with tarfile.open(OUTPATH, "r") as tf:
            members = tf.getmembers()
            image_ids = ["-".join(member.name.split("-")[:-2]) for member in members]
    
    for protein_name, protein_images in tqdm(metadata.items(), desc="Proteins"):
        with tarfile.open(OUTPATH, "a") as tf:
            for info in tqdm(protein_images, desc="Images", leave=False):
                if info["image-id"] in image_ids:
                    continue
                
                # Updates metadata if needed
                info["protein-id"] = protein_name

                # Reads image
                if info["image-type"] == "msr":
                    with MSRReader() as msrreader:
                        image = msrreader.read(os.path.join(BASEPATH, info["image-id"]))
                elif info["image-type"] == "npz":
                    image = numpy.load(os.path.join(BASEPATH, info["image-id"]))
                else:
                    image = tifffile.imread(os.path.join(BASEPATH, info["image-id"]))

                # Indexes image
                if not isinstance(info["chan-id"], type(None)):
                    try:
                        image = image[info["chan-id"]]
                    except Exception as err:
                        print("\n")
                        print("`chan-id` not found in file")
                        if info["image-type"] == "msr":
                            print(image.keys())
                        print(info)
                        continue

                # If a side of image is smaller than CROP_SIZE we remove
                if (image.shape[-2] < CROP_SIZE) or (image.shape[-1] < CROP_SIZE):
                    continue
                
                # Min-Max normalization
                m, M = numpy.quantile(image, [0.01, 0.99])
                if m == M: 
                    print("\n")
                    print("Min-Max normalization impossible... Skipping")
                    print(info)
                    print(f"{numpy.min(image)=}, {numpy.max(image)=}")
                    continue
                image = (image - m) / (M - m) * 255
                image = image.astype(numpy.uint8)

                # Calculates forrground from Otsu
                threshold = filters.threshold_otsu(image)
                foreground = image > threshold

                for j in range(0, image.shape[-2], CROP_SIZE):
                    for i in range(0, image.shape[-1], CROP_SIZE):
                        # NOTE. This generates the edge crops on right/bottom
                        slc = (
                            slice(j, j + CROP_SIZE) if j + CROP_SIZE < image.shape[-2] else slice(image.shape[-2] - CROP_SIZE, image.shape[-2]),
                            slice(i,  i + CROP_SIZE) if i + CROP_SIZE < image.shape[-1] else slice(image.shape[-1] - CROP_SIZE, image.shape[-1]),
                        )
                        foreground_crop = foreground[slc]
                        if foreground_crop.sum() > MINIMUM_FOREGROUND * CROP_SIZE ** 2:
                            image_crop = image[slc]

                            buffer = io.BytesIO()
                            numpy.savez(buffer, image=image_crop, metadata=info)
                            buffer.seek(0)

                            tarinfo = tarfile.TarInfo(name=f'{info["image-id"]}-{j}-{i}')
                            tarinfo.size = len(buffer.getbuffer())
                            tf.addfile(tarinfo=tarinfo, fileobj=buffer)
            
if __name__ == "__main__":
    
    try:
        main()
    except Exception as err:
        javabridge.kill_vm()
        raise err
    javabridge.kill_vm()