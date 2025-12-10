
import os
import glob
import tarfile
import numpy
import io
import tiffwrapper
import SimpleITK as sitk
import tifffile

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from skimage.filters import threshold_otsu

from stedfm.DEFAULTS import BASE_PATH

# from mureader import Reader

MSRKEYS = [
    ("STED_561 {16}", "STED_561 {15}"),
    ("STED_640 {16}", "STED_640 {15}"),
    ("bad_STED 561 {13}", "STED 561 {11}"),
    ("bad_STED 640 {13}", "STED 640 {11}"),
    ("BADSTED_561 {16}", "STED_561 {15}"),
    ("BADSTED_640 {16}", "STED_640 {15}"),
]

CROP_SIZE = 224
THRESHOLD = 0.0

OUTPATH = os.path.join(BASE_PATH, "denoising-data", "kt-lqhq")

def register_images(stack):

    stack = stack.astype(numpy.float32)

    moving_image = sitk.GetImageFromArray(stack[0])
    fixed_image = sitk.GetImageFromArray(stack[1])

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsRegularStepGradientDescent(5.0, 0.001, 200, 0.5, 1e-6)
    R.SetInitialTransform(sitk.TranslationTransform(fixed_image.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)
    outTx = R.Execute(fixed_image, moving_image)

    print("-------")
    print(outTx)
    print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
    print(" Iteration: {0}".format(R.GetOptimizerIteration()))
    print(" Metric value: {0}".format(R.GetMetricValue()))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(outTx)
    out = resampler.Execute(moving_image)

    registered_image = sitk.GetArrayFromImage(out)

    return numpy.stack([registered_image, stack[1]], axis=0)

def normalize(img: numpy.ndarray, channel: int=1):
    """
    Normalize the image to [0, 1] based on the 0.0001 and 0.9999 quantiles.
    Both channels are normalized using the same values.

    :param img: Image to be normalized.
    :param channel: Channel to be used for normalization. Defaults to 1.
    
    :return: Normalized image.
    """
    m, M = numpy.quantile(img, 0.000, keepdims=True, axis=(1, 2)), numpy.quantile(img, 1.0, keepdims=True, axis=(1, 2))
    img = (img - m) / (M - m)
    img = numpy.clip(img, 0, 1)
    img = img.astype(numpy.float32)
    return img

def is_shape_match(data, keys):
    shape = data[keys[0]].shape
    for key in keys[1:]:
        if data[key].shape != shape:
            return False
    return True

def add_files_to_tar(condition, files, split):

    with tarfile.open(os.path.join(OUTPATH, f"{split}-dataset.tar"), "a") as handle:

        start_length = len(handle.getnames())

        for i, f in enumerate(tqdm(files, desc=f"{condition} ({split}) files...")):
            
            good_data = tifffile.imread(f)
            bad_data = tifffile.imread(f.replace("Good_STED", "Bad_STED"))

            img = numpy.stack(
                [bad_data, good_data], axis=0
            )
            # Normalize the image
            img = normalize(img)
            img = register_images(img)

            # Mask is applied on the HQ channel
            mask = img[1] > threshold_otsu(img[1])

            num_y = numpy.floor(img.shape[-2] / CROP_SIZE)
            num_x = numpy.floor(img.shape[-1] / CROP_SIZE)
            ys = numpy.arange(0, num_y*CROP_SIZE, CROP_SIZE).astype(numpy.int64)
            xs = numpy.arange(0, num_x*CROP_SIZE, CROP_SIZE).astype(numpy.int64)

            for y in ys:
                for x in xs:
                    crop = img[:, y:y+CROP_SIZE, x:x+CROP_SIZE]
                    mask_crop = mask[y:y+CROP_SIZE, x:x+CROP_SIZE]

                    foreground = numpy.count_nonzero(mask_crop)
                    pixels = crop.shape[-2] * crop.shape[-1]
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
                        numpy.savez(buffer, image=crop, metadata={"condition": condition})
                        buffer.seek(0)
                        name = f"{condition}-{os.path.basename(f)}-{x}-{y}"

                        tarinfo = tarfile.TarInfo(name=name)
                        tarinfo.size = len(buffer.getbuffer())
                        handle.addfile(tarinfo=tarinfo, fileobj=buffer)

        end_length = len(handle.getnames())
        print(f"Added {end_length - start_length} images to the tar file.")        

def export_to_tiff(condition, files, split):

    os.makedirs(os.path.join(OUTPATH, "tiff-exports", condition, split), exist_ok=True)

    for i, f in enumerate(tqdm(files, desc=f"{condition} ({split}) files...")):

        good_data = tifffile.imread(f)
        bad_data = tifffile.imread(f.replace("Good_STED", "Bad_STED"))

        img = numpy.stack(
            [bad_data, good_data], axis=0
        )
        # Normalize the image
        img = normalize(img)
        img = register_images(img)

        tiffwrapper.imsave(
            os.path.join(OUTPATH, "tiff-exports", condition, split, f"{condition}-{os.path.basename(f)}"),
            img.astype(numpy.float32),
            composite=True,
            luts=["green", "magenta"]
        )


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--export-to-tiff", action="store_true")
    args = parser.parse_args()

    groups = {
        "Gephyrin_STARRED" : glob.glob(os.path.join(BASE_PATH, "denoising-data", "kt-lqhq", "raw", "Gephyrin_STARRED", "Good_STED", "*.tif")),
        "VGAT_ATTO490LS" : glob.glob(os.path.join(BASE_PATH, "denoising-data", "kt-lqhq", "raw", "VGAT_ATTO490LS", "Good_STED", "*.tif")),
    }
    for key, values in groups.items():
        print(key, len(values))

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
    print("Doneski!")