
import os
import numpy
import tarfile
import io

from pystackreg import StackReg
from mureader import Reader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from skimage.filters import threshold_otsu

from stedfm.DEFAULTS import BASE_PATH

CHANNELS = {
    "microtubule": 0,
    "histone": 1,
    "tubulin": None,
}
CROP_SIZE = 224
THRESHOLD = 0.01
OUTPATH = os.path.join(BASE_PATH, "denoising-data", "unet-rcan-lqhq")
os.makedirs(OUTPATH, exist_ok=True)

def register_stack(stack):
    """Register a 3D stack (T, H, W) using rigid body model."""
    print("Registering stack...")
    sr = StackReg(StackReg.RIGID_BODY)
    registered_stack = sr.register_transform_stack(stack, reference='first')
    return registered_stack

def normalize(img: numpy.ndarray, channel: int=1):
    """
    Normalize the image to [0, 1] based on the 0.0001 and 0.9999 quantiles.
    Both channels are normalized using the same values.

    :param img: Image to be normalized.
    :param channel: Channel to be used for normalization. Defaults to 1.
    
    :return: Normalized image.
    """
    # m, M = numpy.quantile(img, 0.0001, axis=(-2, -1), keepdims=True), numpy.quantile(img, 0.9999, axis=(-2, -1), keepdims=True)
    m, M = numpy.min(img, axis=(-2, -1), keepdims=True), numpy.max(img, axis=(-2, -1), keepdims=True)
    img = (img - m) / (M - m + 1e-8)
    img = numpy.clip(img, 0, 1)
    img = img.astype(numpy.float32)
    return img

def add_files_to_tar(condition, filename, stack_names, split):
    with tarfile.open(os.path.join(OUTPATH, f"{split}-dataset.tar"), "a") as handle:

        start_length = len(handle.getnames())
        for i, stack_name in enumerate(stack_names):
            with Reader() as reader:
                try:
                    data = reader.read(filename, keys=stack_name)
                    metadata = reader.get_metadata(filename)[stack_name]
                except Exception as e:
                    print(f"Error reading {stack_name} from {filename}: {e}")
                    continue

            try:
                stack = data[stack_name]
            except Exception as e:
                print(f"Error accessing data for {stack_name} from {filename}: {e}")
                continue
            
            if CHANNELS[condition] is not None:
                stack = stack[:, CHANNELS[condition], ...]

            print(f"Processing stack {i+1}/{len(stack_names)}: {stack_name} with shape {stack.shape}...")
            stack = register_stack(stack)

            gt_stack = numpy.sum(stack, axis=0)
            mask = gt_stack > threshold_otsu(gt_stack)

            gt_stack = normalize(gt_stack)
            stack = normalize(stack)

            num_y = numpy.floor(stack.shape[-2] / CROP_SIZE)
            num_x = numpy.floor(stack.shape[-1] / CROP_SIZE)
            ys = numpy.arange(0, num_y*CROP_SIZE, CROP_SIZE).astype(numpy.int64)
            xs = numpy.arange(0, num_x*CROP_SIZE, CROP_SIZE).astype(numpy.int64)
            
            for t in tqdm(range(stack.shape[0]), desc=f"Stack indexing"):
                for y in ys:
                    for x in xs:
                        crop = stack[t, y:y+CROP_SIZE, x:x+CROP_SIZE]
                        gt_crop = gt_stack[y:y+CROP_SIZE, x:x+CROP_SIZE]
                        mask_crop = mask[y:y+CROP_SIZE, x:x+CROP_SIZE]
                        foreground = numpy.count_nonzero(mask_crop) / mask_crop.size
                        if foreground < THRESHOLD:
                            continue
                        
                        buffer = io.BytesIO()
                        image = numpy.stack([crop, gt_crop], axis=0)
                        numpy.savez(buffer, image=image, metadata={
                            "condition" : condition,
                            "metadata" : metadata,
                        })
                        buffer.seek(0)

                        name = f"{condition}-{os.path.basename(filename)}-{t}-{y}-{x}"
                        tarinfo = tarfile.TarInfo(name=name)
                        tarinfo.size = len(buffer.getbuffer())
                        handle.addfile(tarinfo=tarinfo, fileobj=buffer)

        end_length = len(handle.getnames())
        print(f"Added {end_length - start_length} samples to the {split} dataset.")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--export-to-tiff", action="store_true")
    args = parser.parse_args()

    groups = {
        "microtubule": os.path.join(BASE_PATH, "denoising-data", "unet-rcan-lqhq-mt-hist", "raw", "Microtubule580-histon-2D-002.lif"),
        "histone": os.path.join(BASE_PATH, "denoising-data", "unet-rcan-lqhq-mt-hist", "raw", "Microtubule580-histon-2D-002.lif"),
        "tubulin": os.path.join(BASE_PATH, "denoising-data", "unet-rcan-lqhq-tub", "raw", "Tubulin-11292021-001.lif"),
    }

    if args.overwrite:
        for split in ["train", "valid", "test"]:
            if os.path.exists(os.path.join(OUTPATH, f"{split}-dataset.tar")):
                os.remove(os.path.join(OUTPATH, f"{split}-dataset.tar"))

    for group, filename in groups.items():
        print(f"Processing {group}...")
        with Reader() as reader:
            metadata = reader.get_metadata(filename)
            valid_stack_names = [name for name, stack_metadata in metadata.items() if stack_metadata['Pixels']['SizeT'] > 1]
        
        training_files, validation_files = train_test_split(valid_stack_names, test_size=0.3, random_state=42)
        validation_files, testing_files = train_test_split(validation_files, test_size=0.5, random_state=42)

        add_files_to_tar(group, filename, training_files, "train")
        add_files_to_tar(group, filename, validation_files, "valid")
        add_files_to_tar(group, filename, testing_files, "test")

if __name__ == "__main__":
    main()