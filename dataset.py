
import os 
import json
import tarfile
import numpy 
import io
import javabridge
import tifffile
import argparse 
import logging 
import uuid

from tqdm.auto import tqdm
from skimage import filters

from utils.msrreader import MSRReader

def position_iterator(image, crop_size=224):
    """
    Generates positions for crops in an image.
    
    Args:
        image (numpy.ndarray): The input image.
        crop_size (int): The size of the crops.
        
    Yields:
        tuple: A tuple containing the row and column indices for each crop.
    """
    for j in range(0, image.shape[-2], crop_size):
        for i in range(0, image.shape[-1], crop_size):
            yield j, i

def from_tarfile():

    BASEPATH = "/home-local2/projects/FLCDataset"
    OUTPATH = "/home-local2/projects/FLCDataset/20250523-subset-dataset-crops.tar"
    CROP_SIZE = 224
    MINIMUM_FOREGROUND = 0.001

    logging.basicConfig(
        filename="dataset.log", filemode="w", encoding="utf-8", level=logging.DEBUG,
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

    metadata = json.load(open("./datasets/metadata.json", "r"))
    positions = json.load(open("/home-local2/projects/FLCDataset/positions.json", "r"))

    if args.overwrite:
        with tarfile.open(OUTPATH, "w") as tf:
            image_ids = []
    else:
        with tarfile.open(OUTPATH, "r") as tf:
            members = tf.getmembers()
            image_ids = ["-".join(member.name.split("-")[:-2]) for member in members]
    
    total_crops = 0
    image_ids = set()
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
                
                try:
                    if info["image-type"] == "msr":
                        with MSRReader() as msrreader:
                            image = msrreader.read(os.path.join(BASEPATH, info["image-id"]))
                    elif info["image-type"] == "npz":
                        image = numpy.load(os.path.join(BASEPATH, info["image-id"]))
                    else:
                        image = tifffile.imread(os.path.join(BASEPATH, info["image-id"]))
                except Exception as err:
                    logging.info("ImageReadError")
                    logging.info(f"Error reading image {info['image-id']}: {err}")
                    logging.info(f"{info=}")
                    continue

                if "chan-id" not in info:
                    logging.info("ChannelIDNotFoundError")
                    logging.info("`chan-id` not found in metadata")
                    logging.info(f"{info=}")
                    continue

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
                m, M = numpy.quantile(image, [0.001, 0.999])
                if m == M: 
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

                # Gets a set of predefined positions for an image
                bypass = True
                pos_iter = positions.get(info["image-id"], None)
                if pos_iter is None:
                    logging.info("PositionNotFoundError")
                    logging.info("No positions found for this image")
                    logging.info(f"{info=}")
                    pos_iter = position_iterator(image, crop_size=CROP_SIZE)
                    bypass = False
                else:
                    pos_iter = list(set([(j, i) for j, i in pos_iter])) # Remove duplicates

                # Anonymization
                name = str(uuid.uuid3(uuid.NAMESPACE_DNS, info["image-id"])) 
                info["image-id"] = name
                while info["image-id"] in image_ids:
                    logging.info("ImageAlreadyExists Collision")
                    name = str(uuid.uuid3(uuid.NAMESPACE_DNS, info["image-id"] + str(hash(info["image-id"]))))
                    info["image-id"] = name
                image_ids.add(info["image-id"])

                ###############################
                # Using crops 
                ###############################


                # Calculates foreground from Otsu
                threshold = filters.threshold_otsu(image_uint8)
                foreground = image_uint8 > threshold

                for j, i in pos_iter:
                    # NOTE. This generates the edge crops on right/bottom
                    slc = (
                        slice(j, j + CROP_SIZE) if j + CROP_SIZE < image.shape[-2] else slice(image.shape[-2] - CROP_SIZE, image.shape[-2]),
                        slice(i,  i + CROP_SIZE) if i + CROP_SIZE < image.shape[-1] else slice(image.shape[-1] - CROP_SIZE, image.shape[-1]),
                    )
                    foreground_crop = foreground[slc]
                    if bypass or foreground_crop.sum() > MINIMUM_FOREGROUND * CROP_SIZE ** 2:
                        image_crop = image[slc]

                        buffer = io.BytesIO()
                        name = f'{info["protein-id"]}-{info["image-id"]}-{j}-{i}'
                        if args.saveastiff:
                            name += ".tif"
                            tifffile.imwrite(buffer, image_crop, imagej=True)
                        else:
                            numpy.savez(buffer, image=image_crop, metadata=info)

                        # buffer = io.BytesIO()
                        # numpy.savez(buffer, image=image_crop, metadata=info)
                        buffer.seek(0)
                        tarinfo = tarfile.TarInfo(name=name)
                        tarinfo.size = len(buffer.getbuffer())
                        tf.addfile(tarinfo=tarinfo, fileobj=buffer)

                        total_crops += 1
                        if total_crops % 1000 == 0:
                            logging.info(f"{total_crops=}")

def from_datasets_subset_metadata():

    BASEPATH = "/home-local2/projects/FLCDataset"
    OUTPATH = "/home-local2/projects/FLCDataset/20250523-subset-dataset-crops.tar"
    CROP_SIZE = 224
    MINIMUM_FOREGROUND = 0.001

    logging.basicConfig(
        filename="dataset.log", filemode="w", encoding="utf-8", level=logging.DEBUG,
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

    metadata = json.load(open("./datasets/metadata.json", "r"))

    if args.overwrite:
        with tarfile.open(OUTPATH, "w") as tf:
            image_ids = []
    else:
        with tarfile.open(OUTPATH, "r") as tf:
            members = tf.getmembers()
            image_ids = ["-".join(member.name.split("-")[:-2]) for member in members]
    
    total_crops = 0
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
                
                try:
                    if info["image-type"] == "msr":
                        with MSRReader() as msrreader:
                            image = msrreader.read(os.path.join(BASEPATH, info["image-id"]))
                    elif info["image-type"] == "npz":
                        image = numpy.load(os.path.join(BASEPATH, info["image-id"]))
                    else:
                        image = tifffile.imread(os.path.join(BASEPATH, info["image-id"]))
                except Exception as err:
                    logging.info("ImageReadError")
                    logging.info(f"Error reading image {info['image-id']}: {err}")
                    logging.info(f"{info=}")
                    continue

                if "chan-id" not in info:
                    logging.info("ChannelIDNotFoundError")
                    logging.info("`chan-id` not found in metadata")
                    logging.info(f"{info=}")
                    continue

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
                m, M = numpy.quantile(image, [0.001, 0.999])
                if m == M: 
                    logging.info("InvalidNormalizationError")
                    logging.info("Min-Max normalization impossible... Skipping")
                    logging.info(f"{info=}")
                    logging.info(f"{numpy.min(image)=}, {numpy.max(image)=}")
                    continue

                image_uint8 = numpy.clip((image - m) / (M - m), 0, 1) * 255
                image_uint8 = image.astype(numpy.uint8)
                if not args.raw:
                    image = image_uint8.copy()
                else:
                    image = image.astype(numpy.uint16)
                    if image.min() != 0:
                        image = image - image.min()   

                # Anonymization
                name = str(uuid.uuid3(uuid.NAMESPACE_DNS, info["image-id"])) 
                info["image-id"] = name

                ###############################
                # Using crops 
                ###############################

                # Calculates foreground from Otsu
                threshold = filters.threshold_otsu(image_uint8)
                foreground = image_uint8 > threshold

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
                            name = f'{info["protein-id"]}-{info["image-id"]}-{j}-{i}'
                            if args.saveastiff:
                                name += ".tif"
                                tifffile.imwrite(buffer, image_crop, imagej=True)
                            else:
                                numpy.savez(buffer, image=image_crop, metadata=info)

                            # buffer = io.BytesIO()
                            # numpy.savez(buffer, image=image_crop, metadata=info)
                            buffer.seek(0)
                            tarinfo = tarfile.TarInfo(name=name)
                            tarinfo.size = len(buffer.getbuffer())
                            tf.addfile(tarinfo=tarinfo, fileobj=buffer)

                            total_crops += 1
                            if total_crops % 1000 == 0:
                                logging.info(f"{total_crops=}")

def main():

    BASEPATH = "/home-local2/projects/FLCDataset"
    OUTPATH = "/home-local2/projects/FLCDataset/20240718-dataset-full-images.tar"
    OUTPATH = "/home-local2/projects/FLCDataset/20250522-dataset-crops.tar"
    # OUTPATH = "/home-local2/projects/FLCDataset/tmp.tar"
    CROP_SIZE = 224
    MINIMUM_FOREGROUND = 0.001

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

    metadata = json.load(open("./datasets/scraping/metadata-updated.json", "r"))

    if args.overwrite:
        with tarfile.open(OUTPATH, "w") as tf:
            image_ids = []
    else:
        with tarfile.open(OUTPATH, "r") as tf:
            members = tf.getmembers()
            image_ids = ["-".join(member.name.split("-")[:-2]) for member in members]       

    image_ids = set()

    total_crops = 0
    with tarfile.open(OUTPATH, "a") as tf:
        for key, info in tqdm(metadata.items(), desc="Images"):
            if info["image-id"] in image_ids:
                continue

            info["chan-id"] = info["msr-key"]
            info["protein-id"] = "unknown"

            image = tifffile.imread(info["path"])

            # Updates metadata if needed; Anonymization                
            # print(f"Processing {info['image-id']}")
            # print(f"key: {info['key']}")
            name = str(uuid.uuid3(uuid.NAMESPACE_DNS, info["image-id"])) 
            info["image-id"] = name
            del info["path"]
            del info["folder"]

            while info["image-id"] in image_ids:
                logging.info("ImageAlreadyExists Collision")
                name = str(uuid.uuid3(uuid.NAMESPACE_DNS, info["image-id"] + str(hash(info["image-id"]))))
                info["image-id"] = name
            image_ids.add(info["image-id"])

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
            
            # buffer = io.BytesIO()
            # numpy.savez(buffer, image=image, metadata=info)
            # buffer.seek(0)

            # tarinfo = tarfile.TarInfo(name=f'{info["image-id"]}')
            # tarinfo.size = len(buffer.getbuffer())
            # tf.addfile(tarinfo=tarinfo, fileobj=buffer)   
            # total_crops += 1
            # if total_crops % 100 == 0:
            #     logging.info(f"{total_crops=}")   

            ################################
            # Using crops 
            ################################                  

            # Calculates foreground from Otsu
            threshold = filters.threshold_otsu(image_uint8)
            foreground = image_uint8 > threshold

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
                        name = f'{info["image-id"]}-{j}-{i}'
                        if args.saveastiff:
                            name += ".tif"
                            unit = info['msr-metadata']['PhysicalSizeXUnit']
                            resolution = (1/info['msr-metadata']['PhysicalSizeX'], 1/info['msr-metadata']['PhysicalSizeY'])
                            if unit in ("m"):
                                unit = "um"
                                resolution = (resolution[0] * 1e-6, resolution[1] * 1e-6) 
                            elif unit in ("µm", "um", "micrometer", "micrometers"):
                                unit = "um"
                                pass
                            elif unit in ("nm", "nanometer", "nanometers"):
                                unit = "um"
                                resolution = (resolution[0] / 1000, resolution[1] / 1000)
                            else:
                                unit = None
                            tifffile.imwrite(buffer, image_crop, imagej=True, 
                                             resolution=resolution, metadata={"unit" : unit})
                        else:
                            numpy.savez(buffer, image=image_crop, metadata=info)
                        buffer.seek(0)

                        tarinfo = tarfile.TarInfo(name=name)
                        tarinfo.size = len(buffer.getbuffer())
                        tf.addfile(tarinfo=tarinfo, fileobj=buffer)    
                        total_crops += 1
                        if total_crops % 1000 == 0:
                            logging.info(f"{total_crops=}")       
    print(f"Total crops: {total_crops}")
   
if __name__ == "__main__":
    
    try:
        main()
        # from_datasets_subset_metadata()
        # from_tarfile()
    except Exception as err:
        javabridge.kill_vm()
        raise err
    javabridge.kill_vm()