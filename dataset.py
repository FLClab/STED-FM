
import os 
import json
import tarfile
import numpy 
import io
import javabridge
import tifffile

from tqdm.auto import tqdm

from utils.msrreader import MSRReader

BASEPATH = "/home-local2/projects/FLCDataset"
OUTPATH = "/home-local2/projects/FLCDataset/dataset.tar"
CROP_SIZE = 224
MINIMUM_FOREGROUND = 0.01

def main():
    metadata = json.load(open("./datasets/metadata.json", "r"))

    with tarfile.open(OUTPATH, "w") as tf:
        pass
    
    for protein_name, protein_images in tqdm(metadata.items(), desc="Proteins"):
        for info in tqdm(protein_images, desc="Images", leave=False):
            if info["image-type"] == "msr":
                with MSRReader() as msrreader:
                    out = msrreader.read(os.path.join(BASEPATH, info["image-id"]))
                    image = out[info["chan-id"]]
                break
            else:
                image = tifffile.imread(os.path.join(BASEPATH, info["image-id"]))
                image = image[info["chan-id"]]

            m, M = numpy.quantile(image, [0.01, 0.99])
            image = (image - m) / (M - m)
            image = image.astype(numpy.float32)

            threshold = numpy.quantile(image, 0.75)
            foreground = image > threshold
            for j in range(0, image.shape[-2] - CROP_SIZE, CROP_SIZE):
                for i in range(0, image.shape[-1] - CROP_SIZE, CROP_SIZE):
                    slc = (
                        slice(j, j + CROP_SIZE),
                        slice(i,  i + CROP_SIZE)
                    )
                    foreground_crop = foreground[slc]
                    if foreground_crop.sum() > MINIMUM_FOREGROUND * CROP_SIZE ** 2:
                        image_crop = image[slc]

                        with tarfile.open(OUTPATH, "a") as tf:
                            buffer = io.BytesIO()
                            numpy.savez(buffer, image=image_crop, metadata=info)
                            buffer.seek(0)

                            tarinfo = tarfile.TarInfo(name=f'{info["image-id"]}-{j}-{i}')
                            tarinfo.size = len(buffer.getbuffer())
                            tf.addfile(tarinfo=tarinfo, fileobj=buffer)

            break
        
            
if __name__ == "__main__":
    
    try:
        main()
    except Exception as err:
        javabridge.kill_vm()
        raise err
    javabridge.kill_vm()