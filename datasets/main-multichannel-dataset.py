
import os 
import json
import tarfile
import numpy 
import io
import tifffile
import argparse 
import logging 
import uuid

from tqdm.auto import tqdm
from skimage import filters

def filter_metadata(metadata, filtered_images):
    filtered_metadata = {}
    for key, info in metadata.items():
        key = key + ".tif"
        if key in filtered_images:
            filtered_metadata[key] = info
    return filtered_metadata

def main():

    BASEPATH = "/home-local2/projects/FLCDataset"
    OUTPATH = "/home-local2/projects/FLCDataset/STED-FM-dataset-full-images-multichannel-sted.tar"

    logging.basicConfig(
        filename="dataset-other.log", filemode="w", encoding="utf-8", level=logging.DEBUG,
        format='%(asctime)s %(message)s', datefmt='[%Y%m%d-%H%M%S]'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Overwrites the tar file")
    parser.add_argument("--saveastiff", action="store_true", help="Saves as tiff")
    parser.add_argument("--raw", action="store_true", help="Uses raw images")
    args = parser.parse_args()

    if args.saveastiff:
        OUTPATH = OUTPATH.replace(".tar", "-tiff.tar")
    if args.raw:
        OUTPATH = OUTPATH.replace(".tar", "-raw.tar")

    metadata = json.load(open(os.path.join(BASEPATH, "scraping-multichannel-sted/metadata.json"), "r"))
    filtered_images = json.load(open(os.path.join(BASEPATH, "scraping-multichannel-sted/filtered_multichannel_images.json"), "r"))
    metadata = filter_metadata(metadata, filtered_images)
    print(f"Total images in filtered metadata: {len(metadata)}")

    if args.overwrite or not os.path.exists(OUTPATH):
        with tarfile.open(OUTPATH, "w") as tf:
            image_ids = []
    else:
        with tarfile.open(OUTPATH, "r") as tf:
            members = tf.getmembers()
            image_ids = ["-".join(member.name.split("-")[:-2]) for member in members]       

    image_ids = set(image_ids)

    total_crops = 0
    with tarfile.open(OUTPATH, "a") as tf:
        for key, info in tqdm(metadata.items(), desc="Images"):

            image = tifffile.imread(os.path.join(BASEPATH, "scraping-multichannel-sted", key))

            if info["image-id"] in image_ids:
                continue

            info["chan-id"] = info["msr-key"]
            info["protein-id"] = "unknown"

            # Updates metadata if needed; Anonymization                
            # print(f"Processing {info['image-id']}")
            # print(f"key: {info['key']}")
            # name = str(uuid.uuid3(uuid.NAMESPACE_DNS, info["image-id"])) 
            info["image-id"] = key.replace(".tif", "")

            while info["image-id"] in image_ids:
                logging.info("ImageAlreadyExists Collision")
                name = str(uuid.uuid3(uuid.NAMESPACE_DNS, info["image-id"] + str(hash(info["image-id"]))))
                info["image-id"] = name
            image_ids.add(info["image-id"])
            
            # Make sure that images with >4 channels are skipped
            if image.ndim > 2 and image.shape[0] > 4:
                logging.info("TooManyChannelsError")
                logging.info(f"{info=}")
                logging.info(f"{image.shape=}")
                continue

            # Min-Max normalization
            m, M = numpy.quantile(image, [0.001, 0.999], keepdims=True)
            if any(m == M): 
                logging.info("InvalidNormalizationError")
                logging.info("Min-Max normalization impossible... Skipping")
                logging.info(f"{info=}")
                logging.info(f"{numpy.min(image)=}, {numpy.max(image)=}")
                continue

            image_uint8 = numpy.clip((image - m) / (M - m), 0, 1) * 255
            image_uint8 = image_uint8.astype(numpy.uint8)
            if not args.raw:
                image = image_uint8.copy()
            else:
                image = image.astype(numpy.uint16)
                if image.min() != 0:
                    image = image - image.min()

            ################################
            # Using complete images
            ################################
            
            buffer = io.BytesIO()
            numpy.savez(buffer, image=image, metadata=info)
            buffer.seek(0)

            tarinfo = tarfile.TarInfo(name=f'{info["image-id"]}')
            tarinfo.size = len(buffer.getbuffer())
            tf.addfile(tarinfo=tarinfo, fileobj=buffer)   
            total_crops += 1
            if total_crops % 100 == 0:
                logging.info(f"{total_crops=}")   

    print(f"Total crops: {total_crops}")
   
if __name__ == "__main__":
    
    main()