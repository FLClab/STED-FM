
import os, glob

from packaging import metadata
import tifffile
import uuid 
import pandas
import hashlib
import numpy
import json
import re
import logging

from typing import Generator, List
from collections import defaultdict
from tqdm.auto import tqdm

# import sys
# sys.path.insert(0, "../..")
# from utils.msrreader import MSRReader

# import mureader
# from mureader import Reader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG, filename="output2.log", filemode="w")
logger = logging.getLogger(__name__)

from neurofmdb.parsers.base_parser import get_parser_for_file

DEFAULTPATHS = {
    "pdk-nas" : os.path.expanduser("~/mnt/pdk-nas"),
    "flclab-abberior-sted" : os.path.expanduser("~/valeria-s3/flclab-abberior-sted"),
    "flclab-public" : os.path.expanduser("~/valeria-s3/flclab-public"),
}
MIN_IMAGE_SIZE = 128

def get_hash(string:str):
    return hashlib.sha256(string.encode("utf-8")).hexdigest()

def get_msrfiles(path: str) -> List[str]:
    """
    Gets the list of MSR files from the path.
    """
    return glob.glob(os.path.join(path, "**/*.msr"), recursive=True)

def yield_msrfiles(msrfiles: List=None, outdata: dict=None) -> Generator[str, None, None]:
    """
    Yields the list of MSR files from the path.

    :param path: A `str` of the path to the folder

    :returns : A `str` of the path to the MSR file
    """
    if msrfiles is not None:
        i = 0
        for line in tqdm(msrfiles, desc="Reading MSR files list", total=len(msrfiles)):
            if line.startswith("#") or line.strip() == "":
                continue
            if line.startswith("./"):
                 line = line[2:]
            if "pdk-nas" in line:
                if "#snapshot" in line:
                    continue
                yield os.path.join(os.path.expanduser("~"), "mnt", line.strip())
            else:
                yield os.path.join(os.path.expanduser("~"), "valeria-s3", line.strip())
            i += 1
        return

    path = "."
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
    # Filters : minimum image size; 2D image and multichannel 2D
    return {
        key : value for key, value in image.items()
            if value.ndim in (2, 3)
            and value.shape[-2] > MIN_IMAGE_SIZE 
            and value.shape[-1] > MIN_IMAGE_SIZE
    }

def filter_image_channel(image: dict, filename: str) -> dict:
    """
    Filters the image channel within the dict

    :param image: A `dict` of the image

    :returns : A `dict` of the filtered image
    """
    # Filters: removes overview;
    if filename.endswith(".obf"):
        return {
            key : value for key, value in image.items()
                if "exp" not in key.lower()
                and "focus" not in key.lower()
        }
    else:
        return {
            key : value for key, value in image.items()
                if "overview" not in key.lower()
                and "exp" not in key.lower()
                and "focus" not in key.lower()
        }

def filter_sted_channels(image: dict, filename: str) -> dict:
    """
    Filters the STED channels within the dict

    :param image: A `dict` of the image

    :returns : A `dict` of the filtered image
    """
    if filename.endswith(".obf"):
        return {
            key : value for key, value in image.items()
                if "sted" in key.lower()
        }
    else:
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

def filter_image_keys(image: dict, filename: str) -> dict:
    """
    Filters the image keys within the dict

    :param image: A `dict` of the image

    :returns : A `dict` of the filtered image
    """
    if filename.endswith(".obf"):
        return {
            key.replace(" STED", "") : value for key, value in image.items()
        }
    return image

def handle_metadata(image: dict, metadata: dict, filename: str) -> dict:
    """
    Handles the metadata within the dict

    :param metadata: A `dict` of the metadata

    :returns : A `dict` of the handled metadata
    """
    if filename.endswith(".obf"):
        tmp = defaultdict(list)
        for key in image.keys():
            for candidate_key in metadata.keys():
                if key in candidate_key:
                    remaining = candidate_key.replace(key, "")
                    count = remaining.count("/")
                    if count == 1 and "sted" in remaining.lower():
                        tmp[key].append(metadata[candidate_key])
        metadata = {**metadata, **tmp}
    return metadata

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
    parser.add_argument("--msrfiles", type=str, required=True, nargs="+", help="Path to the text file containing the list of MSR files to process. Can be a space-separated list of files or a single file with one MSR file per line.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--dry-run", action="store_true", help="Dry run")
    args = parser.parse_args()
    
    OUTPUTPATH = "/home-local2/projects/FLCDataset"
    if args.dry_run:
        import tempfile
        OUTPUTPATH = tempfile.gettempdir()
        logger.info(f"Dry run mode: output path is {OUTPUTPATH}")
        outdir = os.path.join(OUTPUTPATH, f"scraping-multichannel-sted-dryrun")
    else:
        outdir = os.path.join(OUTPUTPATH, f"scraping-multichannel-sted")

    if args.msrfiles is not None:
        for msrfile in args.msrfiles:
            if not os.path.isfile(msrfile):
                raise ValueError(f"MSR file {msrfile} does not exist")
    if args.msrfiles:
        outdir = os.path.join(OUTPUTPATH, f"scraping-multichannel-sted")
    
    if args.overwrite:
        if os.path.isdir(outdir):
            logger.info(f"Overwriting existing directory {outdir}")
            import shutil
            shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)


    outdata = {}
    if os.path.isfile(os.path.join(outdir, "metadata.json")) and not args.overwrite:
        outdata = json.load(open(os.path.join(outdir, "metadata.json"), "r"))
        logger.info(f"Loaded existing metadata with {len(outdata)} entries")

    msrfiles = []
    for msrfile in args.msrfiles:
        with open(msrfile, "r") as f:
            msrfiles += [line.strip() for line in f.readlines()]

    seen_msrfiles = set()
    for key, value in outdata.items():
        msrfile = value["image-id"]
        seen_msrfiles.add(msrfile)

    i = 0
    start_idx = 4860
    for current_file_idx, msrfile in enumerate(yield_msrfiles(msrfiles=msrfiles[start_idx:], outdata=outdata)):
        
        if current_file_idx % 10 == 0:
            logger.info(f"Processing file {current_file_idx + start_idx + 1}/{len(msrfiles) if msrfiles else 'unknown'}: {msrfile}")

        try:

            if msrfile in seen_msrfiles:
                logger.info(f"Skipping {msrfile} as it has already been processed")
                continue

            # logger.info(f"Processing {msrfile}")

            possibly_cached_file = msrfile.replace(
                os.path.join(os.path.expanduser("~"), "valeria-s3"),
                "/home-local2/tmp"
            )
            if os.path.isfile(possibly_cached_file):
                logger.info(f"Using cached file for {msrfile}")
                parser = get_parser_for_file(possibly_cached_file)
            else:
                parser = get_parser_for_file(msrfile)
            result = parser.parse()

            image = result["data"]
            metadata = result["other-metadata"]
            # logger.info(metadata.keys())
            if "to_be_merged_keys" in result:
                for keys in result["to_be_merged_keys"]:
                    joined_keys = "/".join(keys)
                    metadata[joined_keys] = [metadata[key] for key in keys]

            # logger.info("Original image keys:", list(image.keys()))

            initial_keys = list(image.keys())
            logger.debug(f"Initial channels in {msrfile}: {initial_keys}")

            # Ensure all images are numpy arrays
            image = filter_image_only(image)
            if len(image) == 0:
                logger.info(f"No valid images found in {msrfile}, skipping")
                logger.info(f"\tInitial channels: {initial_keys}")
                logger.info(f"\tCurrent channels: {list(image.keys())}")
                continue
            logger.debug(f"Channels after filtering non-array values: {list(image.keys())}")
            logger.debug(f"Shapes of channels in {msrfile}:)")
            for key, value in image.items():
                logger.debug(f"\t{key}: {value.shape}")

            # Filter image size
            image = filter_image_size(image)
            if len(image) == 0:
                logger.info(f"No valid images found in {msrfile} after size filtering, skipping")
                logger.info(f"\tInitial channels: {initial_keys}")
                logger.info(f"\tCurrent channels: {list(image.keys())}")
                continue
            logger.debug(f"Channels after size filtering: {list(image.keys())}")

            # Remove overview
            image = filter_image_channel(image, msrfile)
            if len(image) == 0:
                logger.info(f"No valid channels found in {msrfile} after filtering, skipping")
                logger.info(f"\tInitial channels: {initial_keys}")
                logger.info(f"\tCurrent channels: {list(image.keys())}")
                continue
            logger.debug(f"Channels after overview filtering: {list(image.keys())}")

            # Keeps only sted images
            image = filter_sted_channels(image, msrfile)
            if len(image) == 0:
                logger.info(f"No STED channels found in {msrfile}, skipping")
                logger.info(f"\tInitial channels: {initial_keys}")
                logger.info(f"\tCurrent channels: {list(image.keys())}")
                continue
            logger.debug(f"Channels after STED filtering: {list(image.keys())}")

            # Removes the trailing part of the keys
            image = filter_image_keys(image, msrfile)

            # logger.info("Filtered image keys:", list(image.keys()))
            # for key, value in image.items():
            #     logger.info(f"{key}: {value.shape}")

            metadata = handle_metadata(image, metadata, msrfile)

            for key, value in image.items():
                if key not in metadata:
                    logger.info(f"Key {key} not found in metadata, creating default metadata")
                    metadata[key] = {
                        "Pixels" : {
                            "SizeX" : int(value.shape[-1]),
                            "SizeY" : int(value.shape[-2]),
                            "SizeC" : int(value.shape[-3]) if value.ndim == 3 else 1,
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
                            "SizeC" : int(value.shape[-3]) if value.ndim == 3 else 1,
                            "PhysicalSizeX" : 1.0,
                            "PhysicalSizeY" : 1.0,
                            "PhysicalSizeZ" : 1.0,
                            "PhysicalSizeXUnit" : "µm",
                            "PhysicalSizeYUnit" : "µm",
                            "PhysicalSizeZUnit" : "µm",
                        }
                    }
                scale_ = 1.0
                if isinstance(metadata[key], list):
                    metadata_ = metadata[key][0]
                else:
                    metadata_ = metadata[key]
                if metadata_["Pixels"]["PhysicalSizeXUnit"] == "µm":
                    scale_ = 1.0
                elif metadata_["Pixels"]["PhysicalSizeXUnit"] == "nm":
                    scale_ = 1e-3
                elif metadata_["Pixels"]["PhysicalSizeXUnit"] == "m":
                    scale_ = 1e+6
                else:
                    logger.info(f"Unknown unit {metadata_['Pixels']['PhysicalSizeXUnit']}, assuming µm")

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
                    resolution=(1. / (float(metadata_["Pixels"]["PhysicalSizeX"]) * scale_), 1. / (float(metadata_["Pixels"]["PhysicalSizeY"]) * scale_)),
                    imagej=True,
                    metadata = {"unit" : "um", "mode" : "composite"}
                )
            
            i += 1
            if i % 10 == 0:
                json.dump(outdata, open(os.path.join(outdir, "metadata.json"), "w"), sort_keys=True, indent=2)

        except Exception as err:
            logger.error(f"Error processing {msrfile}: {err}")
            continue
    
    json.dump(outdata, open(os.path.join(outdir, "metadata.json"), "w"), sort_keys=True, indent=2)

if __name__ == "__main__":
    
    main()
    # try:
    #     main()
    # except Exception as err:
    #     javabridge.kill_vm()
    #     raise err
    # javabridge.kill_vm()
