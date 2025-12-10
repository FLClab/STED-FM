
import os, glob
import javabridge
import tifffile
import uuid 
import pandas
import hashlib
import numpy
import json
import re

from typing import Generator, List
from tqdm.auto import tqdm

# import sys
# sys.path.insert(0, "../..")
# from utils.msrreader import MSRReader

import mureader
from mureader import Reader

DEFAULTPATHS = {
    "pdk-nas" : os.path.expanduser("~/mnt/pdk-nas"),
    "flclab-abberior-sted" : os.path.expanduser("~/valeria-s3/flclab-abberior-sted"),
    "flclab-public" : os.path.expanduser("~/valeria-s3/flclab-public"),
}
MIN_IMAGE_SIZE = 224

def get_hash(string:str):
    return hashlib.sha256(string.encode("utf-8")).hexdigest()

def get_msrfiles(path: str) -> List[str]:
    """
    Gets the list of MSR files from the path.
    """
    return glob.glob(os.path.join(path, "**/*.msr"), recursive=True)

def yield_msrfiles(path: str, msrfiles: str=None, outdata: dict=None) -> Generator[str, None, None]:
    """
    Yields the list of MSR files from the path.

    :param path: A `str` of the path to the folder

    :returns : A `str` of the path to the MSR file
    """
    if msrfiles is not None:
        with open(msrfiles, "r") as f:
            i = 0
            lines = [line for line in f.readlines()]

            if outdata is not None:
                seen_files = [value["image-id"] for value in outdata.values() if "image-id" in value]
                print(f"Filtering {len(lines)} files with existing {len(outdata)} entries...")
                for line in lines:
                    if "pdk-nas" in msrfiles:
                        filename = os.path.join(os.path.expanduser("~"), "mnt", line[2:].strip())
                    else:
                        filename = os.path.join(os.path.expanduser("~"), "valeria-s3", line.strip())
                    if filename in seen_files:
                        lines.remove(line)
                print(f"Remaining {len(lines)} files to process...")

            f.seek(0)  # Reset file pointer to the beginning
            for line in tqdm(lines, desc="Reading MSR files list", total=len(lines)):
                if "pdk-nas" in msrfiles:
                    if "#snapshot" in line:
                        continue
                    yield os.path.join(os.path.expanduser("~"), "mnt", line[2:].strip())
                else:
                    yield os.path.join(os.path.expanduser("~"), "valeria-s3", line.strip())
                i += 1
        return

    for root, dirs, files in os.walk(path):
        print("Current directory:", root)
        files = list(filter(lambda file: file.endswith(".msr"), files))
        if files:
            for file in files:
                yield os.path.join(root, file)

def remove_content_in_braces_and_filter(text):
    """
    Removes the content and the surrounding curly braces, including
    any preceding space, from a string.
    """
    # The regex pattern matches a space, '{', any characters non-greedily, and '}'.
    pattern = r" \{.*?\}"
    
    # Replace the matched pattern with an empty string
    result = re.sub(pattern, "", text)
    
    # .strip() removes any remaining leading/trailing whitespace
    result = result.strip()
    result = result.replace(" ", "")
    result = result.replace("_", "")
    result = result.replace("-", "")
    return result.strip() 

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

def filter_image_only(image: dict) -> dict:
    """
    Filters the image only within the dict

    :param image: A `dict` of the image

    :returns : A `dict` of the filtered image
    """
    return {
        key : value for key, value in image.items()
            if isinstance(value, numpy.ndarray)
    }

def get_merged_stack(key: str, confocal_key: str, keys: List[str], image: dict) -> numpy.ndarray:
    index = keys.index(confocal_key)
    confocal_key = list(image.keys())[index] # Retrieves the original key
    if not image[key].shape == image[confocal_key].shape:
        # print(f"Shape mismatch between {key} and {confocal_key}")
        return None
    return numpy.stack((image[confocal_key], image[key]), axis=0)

def merge_confocal_sted(image: dict) -> dict:
    """
    Merges the confocal and STED channels

    :param image: A `dict` of the image

    :returns : A `dict` of the merged image
    """
    merged = {}
    dict_keys = [remove_content_in_braces_and_filter(key.lower()) for key in image.keys()]
    for key, value in image.items():
        filtered_key = remove_content_in_braces_and_filter(key.lower())
        if "sted" in filtered_key:
            for attempt in ("conf", "confocal"):
                confocal_key = filtered_key.replace("sted", attempt)
                if confocal_key in dict_keys:
                    tmp = get_merged_stack(key, confocal_key, dict_keys, image)
                    if tmp is not None:
                        merged[key] = tmp
                        # print(f"Merged {key} with {confocal_key}", merged[key].shape)
                    break
            else:
                print(f"Could not find confocal channel for {key}")
                print(dict_keys)
    return merged

def main():
    """
    Main function to convert the MSR files to TIFF files
    """
    import argparse
    parser = argparse.ArgumentParser(description="Convert MSR files to TIFF files")
    parser.add_argument("--path", type=str, default="pdk-nas", help="Path to the MSR files")
    parser.add_argument("--msrfiles", type=str, default=None, help="Path to a text file containing the list of MSR files to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--dry-run", action="store_true", help="Dry run")
    args = parser.parse_args()

    if args.path not in DEFAULTPATHS:
        raise ValueError(f"Path {args.path} not in {list(DEFAULTPATHS.keys())}")
    
    OUTPUTPATH = "/home-local2/projects/FLCDataset"
    if args.dry_run:
        import tempfile
        OUTPUTPATH = tempfile.gettempdir()
        print(f"Dry run mode: output path is {OUTPUTPATH}")
        outdir = os.path.join(OUTPUTPATH, f"scraping-confocal-sted-{args.path}-dryrun")
    else:
        outdir = os.path.join(OUTPUTPATH, f"scraping-confocal-sted-{args.path}")

    if args.msrfiles is not None and not os.path.isfile(args.msrfiles):
        raise ValueError(f"MSR files list {args.msrfiles} does not exist")
    if args.msrfiles:
        if "pdk-nas" in args.msrfiles:
            outdir = os.path.join(OUTPUTPATH, f"scraping-confocal-sted-pdk-nas")
        else:
            outdir = os.path.join(OUTPUTPATH, f"scraping-confocal-sted")
    
    os.makedirs(outdir, exist_ok=True)

    outdata = {}
    if os.path.isfile(os.path.join(outdir, "metadata.json")) and not args.overwrite:
        outdata = json.load(open(os.path.join(outdir, "metadata.json"), "r"))
        print(f"Loaded existing metadata with {len(outdata)} entries")

    for msrfile in yield_msrfiles(DEFAULTPATHS[args.path], msrfiles=args.msrfiles, outdata=outdata):

        with Reader() as msrreader:
            try:
                image = msrreader.read(msrfile)
                metadata = msrreader.get_metadata(msrfile)
            except (OSError, javabridge.jutil.JavaException, mureader.reader.JavaBridgeException) as err:
                print(err)
                print("Could not read the file...")
                continue

            # Ensure all images are numpy arrays
            image = filter_image_only(image)

            # Filter image size
            image = filter_image_size(image)

            # Remove overview
            image = filter_image_channel(image)

            # Attempts to merge confocal and sted images
            image = merge_confocal_sted(image)

            # # Keeps only sted images
            # image = filter_sted_channels(image)


            # for key, value in image.items():
            #     print(key, value.shape)
            
            for key, value in image.items():
                if key not in metadata:
                    metadata[key] = {
                        "Pixels" : {
                            "SizeX" : int(value.shape[-1]),
                            "SizeY" : int(value.shape[-2]),
                            "SizeZ" : int(value.shape[-3]) if value.ndim == 3 else 1,
                            "PhysicalSizeX" : 1.0,
                            "PhysicalSizeY" : 1.0,
                            "PhysicalSizeZ" : 1.0,
                            "PhysicalSizeXUnit" : "µm",
                            "PhysicalSizeYUnit" : "µm",
                            "PhysicalSizeZUnit" : "µm",
                        }
                    }

                to_hash = msrfile + key
                hashvalue = get_hash(to_hash)
                while hashvalue in outdata:
                    to_hash += key
                    hashvalue = get_hash(to_hash)

                if isinstance(metadata[key], numpy.ndarray):
                    # Strange case where metadata is an image; happens on old files
                    metadata[key] = {
                        "Pixels" : {
                            "SizeX" : int(value.shape[-1]),
                            "SizeY" : int(value.shape[-2]),
                            "SizeZ" : int(value.shape[-3]) if value.ndim == 3 else 1,
                            "PhysicalSizeX" : 1.0,
                            "PhysicalSizeY" : 1.0,
                            "PhysicalSizeZ" : 1.0,
                            "PhysicalSizeXUnit" : "µm",
                            "PhysicalSizeYUnit" : "µm",
                            "PhysicalSizeZUnit" : "µm",
                        }
                    }
                scale_ = 1.0
                if metadata[key]["Pixels"]["PhysicalSizeXUnit"] == "µm":
                    scale_ = 1.0
                elif metadata[key]["Pixels"]["PhysicalSizeXUnit"] == "nm":
                    scale_ = 1e-3
                elif metadata[key]["Pixels"]["PhysicalSizeXUnit"] == "m":
                    scale_ = 1e+6
                else:
                    print(f"Unknown unit {metadata[key]['Pixels']['PhysicalSizeXUnit']}, assuming µm")

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
                    resolution=(1. / (metadata[key]["Pixels"]["PhysicalSizeX"] * scale_), 1. / (metadata[key]["Pixels"]["PhysicalSizeY"] * scale_)),
                    imagej=True,
                    metadata = {"unit" : "um", "mode" : "composite"}
                )
            
            json.dump(outdata, open(os.path.join(outdir, "metadata.json"), "w"), sort_keys=True, indent=2)


if __name__ == "__main__":
    
    main()
    # try:
    #     main()
    # except Exception as err:
    #     javabridge.kill_vm()
    #     raise err
    # javabridge.kill_vm()
