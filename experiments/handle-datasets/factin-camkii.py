
import os
import glob
import tarfile
import numpy
import io
import tiffwrapper

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from skimage.filters import threshold_otsu

import sys
sys.path.insert(0, "../")

from DEFAULTS import BASE_PATH

sys.path.insert(0, "../..")

from utils.msrreader import MSRReader

MSRKEY = "STED_635P {2}"
MSRKEY = "STED_594 {2}"
MASKKEY = "Conf_488 {2}"
OPTIONALMSRKEYS = ["STED_594 {2}"]
OPTIONALMSRKEYS = ["STED_635P {2}"]
CROP_SIZE = 224
THRESHOLD = 0.05

OUTPATH = os.path.join(BASE_PATH, "evaluation-data", "factin-camkii")

def normalize(img):
    m, M = numpy.quantile(img, 0.0001), numpy.quantile(img, 0.9999)
    img = (img - m) / (M - m)
    img = numpy.clip(img, 0, 1)
    img = img.astype(numpy.float32)
    return img

def add_files_to_tar(condition, files, split):

    with tarfile.open(os.path.join(OUTPATH, f"{split}-dataset.tar"), "a") as handle:

        start_length = len(handle.getnames())

        with MSRReader() as msrreader:

            for i, f in enumerate(tqdm(files, desc=f"{condition} ({split}) files...")):
                
                data = msrreader.read(f)
                metadata = msrreader.get_metadata(f)

                img = data[MSRKEY]

                if MASKKEY is not None:
                    mask_image = data[MASKKEY]
                    mask = mask_image > threshold_otsu(mask_image)
                else:
                    mask = img > threshold_otsu(img)

                # Normalize the image
                img = normalize(img)
                # Optional keys
                optional_data = {key: normalize(data[key]) for key in OPTIONALMSRKEYS}
                optional_metadata = {key: metadata[key] for key in OPTIONALMSRKEYS}

                num_y = numpy.floor(img.shape[0] / CROP_SIZE)
                num_x = numpy.floor(img.shape[1] / CROP_SIZE)
                ys = numpy.arange(0, num_y*CROP_SIZE, CROP_SIZE).astype(numpy.int64)
                xs = numpy.arange(0, num_x*CROP_SIZE, CROP_SIZE).astype(numpy.int64)

                for y in ys:
                    for x in xs:
                        crop = img[y:y+CROP_SIZE, x:x+CROP_SIZE]
                        mask_crop = mask[y:y+CROP_SIZE, x:x+CROP_SIZE]

                        foreground = numpy.count_nonzero(mask_crop)
                        pixels = crop.shape[0] * crop.shape[1]
                        ratio = foreground / pixels
                        if ratio < THRESHOLD:

                            # from matplotlib import pyplot
                            # fig, ax = pyplot.subplots()
                            # ax.set(title=f"Condition: {condition}, Ratio: {ratio:0.2f}")
                            # ax.imshow(crop, cmap="gray", vmax=0.1)
                            # fig.savefig("crop.png")
                            # pyplot.close(fig)
                            # input("Press any key to continue...")

                            continue 
                        else:

                            # from matplotlib import pyplot
                            # fig, ax = pyplot.subplots()
                            # ax.set(title=f"Condition: {condition}, Ratio: {ratio:0.2f}")
                            # ax.imshow(crop, cmap="gray", vmax=numpy.quantile(crop, 0.99))
                            # fig.savefig("crop.png")
                            # pyplot.close(fig)
                            # input("Press any key to continue...")

                            buffer = io.BytesIO()
                            numpy.savez(buffer, image=crop, metadata={"condition": condition, "msr-metadata": metadata[MSRKEY], "optional-metadata": optional_metadata, "optional-data": optional_data})
                            buffer.seek(0)
                            name = f"{condition}-{os.path.basename(f)}-{x}-{y}"

                            tarinfo = tarfile.TarInfo(name=name)
                            tarinfo.size = len(buffer.getbuffer())
                            handle.addfile(tarinfo=tarinfo, fileobj=buffer)

        end_length = len(handle.getnames())
        print(f"Added {end_length - start_length} images to the tar file.")        

def export_to_tiff(condition, files, split):
        
    with MSRReader() as msrreader:

        for i, f in enumerate(tqdm(files, desc=f"{condition} ({split}) files...")):
            
            data = msrreader.read(f)

            img = data[MSRKEY]
            export_image = numpy.stack((img, *[data[key] for key in OPTIONALMSRKEYS]), axis=0)
            if MASKKEY is not None:
                export_image = numpy.concatenate((export_image, data[MASKKEY][numpy.newaxis]), axis=0)
            tiffwrapper.imwrite(f.replace(".msr", ".tif"), export_image.astype(numpy.uint16), 
                                composite=True, luts=["gray", "magenta", "cyan"])

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--export-to-tiff", action="store_true")
    args = parser.parse_args()

    groups = {
        "CTRL" : glob.glob(os.path.join(BASE_PATH, "evaluation-data", "camkii", "*.msr"), recursive=True),
    }
    if args.export_to_tiff:
        for key, values in groups.items():
            export_to_tiff(key, values, "all")
        return
    exit()
    
    groups = {
        "CTRL" : glob.glob(os.path.join(BASE_PATH, "evaluation-data", "factin-camkii", "**/*Block_GFP/*.msr"), recursive=True),
        "shRNA" : glob.glob(os.path.join(BASE_PATH, "evaluation-data", "factin-camkii", "**/*Block_shRNA/*.msr"), recursive=True),
        "RESCUE" : glob.glob(os.path.join(BASE_PATH, "evaluation-data", "factin-camkii", "**/*Block_[rR]escue/*.msr"), recursive=True),
    }
    if args.export_to_tiff:
        for key, values in groups.items():
            export_to_tiff(key, values, "all")
        return 

    if args.overwrite:
        for split in ["train", "valid", "test"]:
            if os.path.exists(os.path.join(OUTPATH, f"{split}-dataset.tar")):
                os.remove(os.path.join(OUTPATH, f"{split}-dataset.tar"))

    for key, values in groups.items():
        
        training_files, validation_files = train_test_split(values, test_size=0.3, random_state=42)
        validation_files, testing_files = train_test_split(validation_files, test_size=0.5, random_state=42)

        add_files_to_tar(key, training_files, "train")
        add_files_to_tar(key, validation_files, "valid")
        add_files_to_tar(key, testing_files, "test")

if __name__ == "__main__":

    main()