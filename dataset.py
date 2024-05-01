
import os 
import json
import tarfile
import numpy 
import io
import javabridge
import tifffile
import argparse 
import logging 

from tqdm.auto import tqdm
from skimage import filters

from utils.msrreader import MSRReader

BASEPATH = "/home-local2/projects/FLCDataset"
OUTPATH = "/home-local2/projects/FLCDataset/20240430-dataset.tar"
CROP_SIZE = 224
MINIMUM_FOREGROUND = 0.001

def from_datasets_original_metadata():

    logging.basicConfig(
        filename="dataset.log", filemode="w", encoding="utf-8", level=logging.DEBUG,
        format='%(asctime)s %(message)s', datefmt='[%Y%m%d-%H%M%S]'
    )

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
            for info in tqdm(protein_images, desc=protein_name, leave=False):
                if info["image-id"] in image_ids:
                    continue
                
                # Updates metadata if needed
                info["protein-id"] = protein_name

                # Reads image
                if not os.path.isfile(os.path.join(BASEPATH, info["image-id"])):
                    logging.info("FileNotFoundError")
                    logging.info(f"{info=}")
                    continue

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
                        logging.info("ChannelNotFoundError")
                        logging.info("`chan-id` not found in file")
                        if info["image-type"] == "msr":
                            logging.info(f"{image.keys()=}")
                        logging.info(f"{info=}")
                        continue

                # If a side of image is smaller than CROP_SIZE we remove
                if (image.shape[-2] < CROP_SIZE) or (image.shape[-1] < CROP_SIZE):
                    continue
                # If image is >2D (e.g. timelapse, volume) we skip
                if image.ndim != 2:
                    continue
                
                # Min-Max normalization
                m, M = numpy.quantile(image, [0.01, 0.995])
                if m == M: 
                    logging.info("InvalidNormalizationError")
                    logging.info("Min-Max normalization impossible... Skipping")
                    logging.info(f"{info=}")
                    logging.info(f"{numpy.min(image)=}, {numpy.max(image)=}")
                    continue
                image = numpy.clip((image - m) / (M - m), 0, 1) * 255
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

def main():

    logging.basicConfig(
        filename="dataset.log", filemode="w", encoding="utf-8", level=logging.DEBUG,
        format='%(asctime)s %(message)s', datefmt='[%Y%m%d-%H%M%S]'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Overwrites the tar file")
    args = parser.parse_args()

    metadata = json.load(open("./datasets/scraping/metadata-updated.json", "r"))

    if args.overwrite:
        with tarfile.open(OUTPATH, "w") as tf:
            image_ids = []
    else:
        with tarfile.open(OUTPATH, "r") as tf:
            members = tf.getmembers()
            image_ids = ["-".join(member.name.split("-")[:-2]) for member in members]       

    total_crops = 0
    for key, info in tqdm(metadata.items(), desc="Images"):
        with tarfile.open(OUTPATH, "a") as tf:
            if info["image-id"] in image_ids:
                continue
                
            info["chan-id"] = info["msr-key"]
            info["protein-id"] = "unknown"

            image = tifffile.imread(info["path"])

            # If a side of image is smaller than CROP_SIZE we remove
            if (image.shape[-2] < CROP_SIZE) or (image.shape[-1] < CROP_SIZE):
                continue
            # If image is >2D (e.g. timelapse, volume) we skip
            if image.ndim != 2:
                continue
            
            # Min-Max normalization
            m, M = numpy.quantile(image, [0.001, 0.999])
            if m == M: 
                logging.info("InvalidNormalizationError")
                logging.info("Min-Max normalization impossible... Skipping")
                logging.info(f"{info=}")
                logging.info(f"{numpy.min(image)=}, {numpy.max(image)=}")
                continue
            image = numpy.clip((image - m) / (M - m), 0, 1) * 255
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
                        total_crops += 1
                        if total_crops % 1000 == 0:
                            logging.info(f"{total_crops=}")          

if __name__ == "__main__":
    
    try:
        main()
    except Exception as err:
        javabridge.kill_vm()
        raise err
    javabridge.kill_vm()