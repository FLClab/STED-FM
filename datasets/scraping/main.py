
import os, glob
import javabridge
import tifffile
import uuid 
import pandas
import hashlib
import numpy
import json

import sys
sys.path.insert(0, "../..")
from utils.msrreader import MSRReader

DEFAULTPATHS = {
    "pdk-nas" : os.path.expanduser("~/mnt/pdk-nas"),
    "flclab-abberior-sted" : os.path.expanduser("~/valeria-s3/flclab-abberior-sted"),
    "flclab-public" : os.path.expanduser("~/mnt/flclab-public"),
}
OUTPUTPATH = "/home-local2/projects/FLCDataset"
MIN_IMAGE_SIZE = 224

def get_hash(string:str):
    return hashlib.sha256(string.encode("utf-8")).hexdigest()

def get_msrfiles(path: str) -> list[str]:
    """
    Gets the list of MSR files from the path.
    """
    return glob.glob(os.path.join(path, "**/*.msr"), recursive=True)

def yield_msrfiles(path: str) -> str:
    """
    Yields the list of MSR files from the path.

    :param path: A `str` of the path to the folder

    :returns : A `str` of the path to the MSR file
    """
    for root, dirs, files in os.walk(path):
        print("Current directory:", root)
        files = list(filter(lambda file: file.endswith(".msr"), files))
        if files:
            for file in files:
                yield os.path.join(root, file)

def filter_image_size(image: dict) -> dict:
    """
    Filters the image size within the dict

    :param image: A `dict` of the image

    :returns : A `dict` of the filtered image
    """
    # Filters : minimum image size; 2D image
    return {
        key : value for key, value in image.items()
            if value.ndim == 2 
            and value.shape[-2] > MIN_IMAGE_SIZE 
            and value.shape[-1] > MIN_IMAGE_SIZE
    }

def filter_image_channel(image: dict) -> dict:
    """
    Filters the image channel within the dict

    :param image: A `dict` of the image

    :returns : A `dict` of the filtered image
    """
    # Filters: removes overview;
    return {
        key : value for key, value in image.items()
            if "overview" not in key.lower()
            and "exp" not in key.lower()
            and "focus" not in key.lower()
    }

def filter_sted_channels(image: dict) -> dict:
    """
    Filters the STED channels within the dict

    :param image: A `dict` of the image

    :returns : A `dict` of the filtered image
    """
    return {
        key : value for key, value in image.items()
            if "sted" in key.lower()
    }

def main():
    """
    Main function to convert the MSR files to TIFF files
    """
    import argparse
    parser = argparse.ArgumentParser(description="Convert MSR files to TIFF files")
    parser.add_argument("--path", type=str, default="pdk-nas", help="Path to the MSR files")
    args = parser.parse_args()

    outdir = os.path.join(OUTPUTPATH, f"scraping-{args.path}")
    os.makedirs(outdir, exist_ok=True)

    outdata = {}
    i = 0
    for msrfile in yield_msrfiles(DEFAULTPATHS[args.path]):
        with MSRReader() as msrreader:
            try:
                image = msrreader.read(msrfile)
                metadata = msrreader.get_metadata(msrfile)
            except (OSError, javabridge.jutil.JavaException) as err:
                print(err)
                print("Could not read the file...")
                continue

            # Filter image size
            image = filter_image_size(image)

            # Remove overview
            image = filter_image_channel(image)

            # Keeps only sted images
            image = filter_sted_channels(image)

            # for key, value in image.items():
            #     print(key, value.shape)
            
            for key, value in image.items():
                hashvalue = get_hash(msrfile + key)
                outdata[hashvalue] = {
                    "image-id" : msrfile,
                    "image-type" : "tif",
                    "chan-id" : None,
                    "protein-id" : "unknown",
                    "msr-key" : key,
                    "msr-metadata" : metadata[key]
                }
                tifffile.imwrite(
                    os.path.join(outdir, f"{hashvalue}.tif"), 
                    value.astype(numpy.uint16),
                    resolution = (1. / (metadata[key]["PhysicalSizeX"] * 1e+6), 1. / (metadata[key]["PhysicalSizeY"] * 1e+6)),
                    imagej=True,
                    metadata = {"unit" : "um"}
                )
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i} files")
                json.dump(outdata, open(os.path.join(outdir, "metadata.json"), "w"), sort_keys=True, indent=2)
            i += 1


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        javabridge.kill_vm()
        raise err
    javabridge.kill_vm()
    