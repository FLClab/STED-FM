
import os 
import glob
import json
import tarfile
import numpy 
import io
import tifffile
import argparse 
import logging 

from tqdm.auto import tqdm
from skimage import filters
import read_mrc

BASEPATH = "/home-local2/projects/SSL/ssl-data/sim"
OUTPATH = "/home-local2/projects/SSL/ssl-data/sim-dataset-full-images.tar"
CROP_SIZE = 224
MINIMUM_FOREGROUND = 0.01

FOLDERS={
    "3DRCAN" : ["**/*_decon.tif", "**/GT/*", "**/gt/*", "**/*(ground truth)*/*"],
    "BioSR" : "**/GT_all.mrc",
    "DeepBacs" : "**/SIM/*",
    "EMTB" : "**/Final_DL_Result.tif",
}

def get_protein_id(folder, filename):
    if folder == "3DRCAN":
        if "Denoising" in filename:
            return os.path.dirname(filename).split("Denoising")[-1][1:].split(os.path.sep)[0]
        elif "Expansion_Microscopy" in filename:
            return os.path.dirname(filename).split("Expansion_Microscopy")[-1][1:].split(os.path.sep)[0]
        elif "live_cell_test_data" in filename:
            return os.path.dirname(filename).split("live_cell_test_data")[-1][1:].split(os.path.sep)[0]
    elif folder == "BioSR":
        return os.path.dirname(filename).split(folder)[-1][1:].split(os.path.sep)[0]
    elif folder == "DeepBacs":
        return os.path.dirname(filename).split(folder)[-1][1:].split(os.path.sep)[0]
    elif folder == "EMTB":
        return "EMTB"
    else:
        return "unknown"

def load_files():
    metadata = {}
    for key, searches in FOLDERS.items():
        if isinstance(searches, str):
            searches = [searches]

        files = []
        for search in searches:
            files += glob.glob(os.path.join(BASEPATH, search), recursive=True)
        
        for file in files:
            if file.endswith(".tif") or file.endswith(".mrc"):
                metadata[file] = {
                    "path" : file,
                    "image-id" : file.split(BASEPATH)[1][1:],
                    "image-type" : os.path.splitext(file)[1][1:],
                    "msr-key" : None,
                    "protein-id" : get_protein_id(key, file),
                }
    return metadata

def main():

    logging.basicConfig(
        filename="dataset.log", filemode="w", encoding="utf-8", level=logging.DEBUG,
        format='%(asctime)s %(message)s', datefmt='[%Y%m%d-%H%M%S]'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Overwrites the tar file")
    args = parser.parse_args()

    metadata = load_files()

    if args.overwrite:
        with tarfile.open(OUTPATH, "w") as tf:
            image_ids = []
    else:
        with tarfile.open(OUTPATH, "r") as tf:
            members = tf.getmembers()
            image_ids = ["-".join(member.name.split("-")[:-2]) for member in members]       

    total_crops = 0
    with tarfile.open(OUTPATH, "a") as tf:
        for key, info in tqdm(metadata.items(), desc="Images"):
            if info["image-id"] in image_ids:
                continue
                
            info["chan-id"] = info["msr-key"]

            if info["image-type"] == "mrc":
                _, image = read_mrc.read_mrc(info["path"])
            elif info["image-type"] == "tif":
                image = tifffile.imread(info["path"])

            # If a side of image is smaller than CROP_SIZE we remove
            if (image.shape[-2] < CROP_SIZE) or (image.shape[-1] < CROP_SIZE):
                continue

            # Calculates forrground from Otsu
            threshold = filters.threshold_otsu(image)
            foreground = image > threshold            
            m, M = numpy.quantile(image, [0.001, 0.999])
            if m == M: 
                logging.info("InvalidNormalizationError")
                logging.info("Min-Max normalization impossible... Skipping")
                logging.info(f"{info=}")
                logging.info(f"{numpy.min(image)=}, {numpy.max(image)=}")
                continue            
            if image.ndim == 2:
                # Min-Max normalization
                image = numpy.clip((image - m) / (M - m), 0, 1) * 255
                image = image.astype(numpy.uint8)

                ################################
                # Using complete images
                ################################

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
                
                # buffer = io.BytesIO()
                # numpy.savez(buffer, image=image, metadata=info)
                # buffer.seek(0)

                # tarinfo = tarfile.TarInfo(name=f'{info["image-id"]}')
                # tarinfo.size = len(buffer.getbuffer())
                # tf.addfile(tarinfo=tarinfo, fileobj=buffer)   
                # total_crops += 1
                # if total_crops % 100 == 0:
                #     logging.info(f"{total_crops=}")   
            else:
                for t, img in enumerate(image):
                    
                    info["chan-id"] = f"{t}"

                    # Min-Max normalization
                    img = numpy.clip((img - m) / (M - m), 0, 1) * 255
                    img = img.astype(numpy.uint8)

                    ################################
                    # Using complete images
                    ################################

                    for j in range(0, img.shape[-2], CROP_SIZE):
                        for i in range(0, img.shape[-1], CROP_SIZE):
                            # NOTE. This generates the edge crops on right/bottom
                            slc = (
                                slice(j, j + CROP_SIZE) if j + CROP_SIZE < img.shape[-2] else slice(img.shape[-2] - CROP_SIZE, img.shape[-2]),
                                slice(i,  i + CROP_SIZE) if i + CROP_SIZE < img.shape[-1] else slice(img.shape[-1] - CROP_SIZE, img.shape[-1]),
                            )
                            foreground_crop = foreground[t][slc]
                            if foreground_crop.sum() > MINIMUM_FOREGROUND * CROP_SIZE ** 2:
                                image_crop = img[slc]

                                buffer = io.BytesIO()
                                numpy.savez(buffer, image=image_crop, metadata=info)
                                buffer.seek(0)

                                tarinfo = tarfile.TarInfo(name=f'{info["image-id"]}-{t}-{j}-{i}')
                                tarinfo.size = len(buffer.getbuffer())
                                tf.addfile(tarinfo=tarinfo, fileobj=buffer)    
                                total_crops += 1
                                if total_crops % 1000 == 0:
                                    logging.info(f"{total_crops=}")         

                    
                    # buffer = io.BytesIO()
                    # numpy.savez(buffer, image=img, metadata=info)
                    # buffer.seek(0)

                    # info["chan-id"] = f"{t}"
                    # tarinfo = tarfile.TarInfo(name=f'{info["image-id"]}-{t}')
                    # tarinfo.size = len(buffer.getbuffer())
                    # tf.addfile(tarinfo=tarinfo, fileobj=buffer)   
                    # total_crops += 1
                    # if total_crops % 100 == 0:
                    #     logging.info(f"{total_crops=}")                   

            ################################
            # Using crops 
            ################################                  

            # for j in range(0, image.shape[-2], CROP_SIZE):
            #     for i in range(0, image.shape[-1], CROP_SIZE):
            #         # NOTE. This generates the edge crops on right/bottom
            #         slc = (
            #             slice(j, j + CROP_SIZE) if j + CROP_SIZE < image.shape[-2] else slice(image.shape[-2] - CROP_SIZE, image.shape[-2]),
            #             slice(i,  i + CROP_SIZE) if i + CROP_SIZE < image.shape[-1] else slice(image.shape[-1] - CROP_SIZE, image.shape[-1]),
            #         )
            #         foreground_crop = foreground[slc]
            #         if foreground_crop.sum() > MINIMUM_FOREGROUND * CROP_SIZE ** 2:
            #             image_crop = image[slc]

            #             buffer = io.BytesIO()
            #             numpy.savez(buffer, image=image_crop, metadata=info)
            #             buffer.seek(0)

            #             tarinfo = tarfile.TarInfo(name=f'{info["image-id"]}-{j}-{i}')
            #             tarinfo.size = len(buffer.getbuffer())
            #             tf.addfile(tarinfo=tarinfo, fileobj=buffer)    
            #             total_crops += 1
            #             if total_crops % 1000 == 0:
            #                 logging.info(f"{total_crops=}")         


if __name__ == "__main__":

    main()