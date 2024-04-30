
import os, glob
import json
import tifffile

from collections import defaultdict
from tqdm.auto import tqdm 

def merge_by_metadata(metadata: dict) -> dict: 
    """
    Merge the metadata

    :param images: A `dict` of the metadata

    :returns : A `dict` of the filtered images
    """
    out = defaultdict(list)
    for key, value in metadata.items():
        channel = value["msr-key"]
        h, w = value["msr-metadata"]["SizeY"], value["msr-metadata"]["SizeX"]

        value["key"] = key
        out[(channel, h, w)].append(value)
    return out

def assert_different_images(images: list[str]) -> list[str]:
    """
    Ensures the images are different

    :param images: A `list` of the images

    :returns : A `list` of singleton images
    """
    arrays = []
    for image in images: 
        arrays.append(tifffile.imread(image["path"]))
    
    # Assume first image is different
    different_images = [
        (images[0], arrays[0])
    ]
    for i in range(1, len(arrays)):
        is_different = True
        for metadata, array in different_images:
            if (arrays[i] == array).all():
                is_different = False
                break
        if is_different:
            different_images.append((images[i], arrays[i]))
    return [metadata for metadata, array in different_images]            

def main():

    PATH = "/home-local2/projects/FLCDataset/"
    folders = [
        "scraping-pdk-nas",
        "scraping-flclab-abberior-sted"
    ]

    merged = {}
    for folder in folders:
        metadata = json.load(open(os.path.join(PATH, folder, "metadata.json"), "r"))
        for key, value in metadata.items():
            value["folder"] = os.path.join(PATH, folder)
            value["path"] = os.path.join(PATH, folder, f"{key}.tif")
        merged.update(metadata)
    
    out = merge_by_metadata(merged)
    print("Before merging:", len(merged))
    print("After merging:", len(out))

    for key, values in tqdm(out.items()):
        if len(values) > 1:
            values = assert_different_images(values)
            # Inplace modification
            out[key] = values
    
    # Convert back to the metadata
    metadata = {}
    for key, values in out.items():
        for value in values:
            metadata[value["key"]] = value
    print("After filtering:", len(metadata))
    json.dump(metadata, open(os.path.join(".", "metadata.json"), "w"), sort_keys=True, indent=2)
    
if __name__ == "__main__":

    main()