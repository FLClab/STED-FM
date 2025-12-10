
import numpy
import SimpleITK as sitk
import tifffile
import os
import glob

from PIL import Image
from stedfm.DEFAULTS import BASE_PATH
from tqdm.auto import tqdm

def register_images(stack):

    stack = stack.astype(numpy.float32)

    moving_image = sitk.GetImageFromArray(stack[0])
    fixed_image = sitk.GetImageFromArray(stack[1])

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsRegularStepGradientDescent(4.0, 0.01, 200)
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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Which dataset to register", choices=["ov-lqhq-mt"])
    args = parser.parse_args()

    if args.dataset == "ov-lqhq-mt":
        path = os.path.join(BASE_PATH, "denoising-data", "ov-lqhq-mt", "fixed_cell_microtubule_u2os_alphatubulin_star635p")

        source_files = glob.glob(os.path.join(path, "**/low_intensity_image_patches/*.png*"), recursive=True)
        target_files = [source.replace("low_intensity_image_patches", "ground_truth_image_patches") for source in source_files]
        
        output_folder = path + "_registered"

    for source_file, target_file in zip(tqdm(source_files), target_files):

        ext = os.path.splitext(source_file)[1].lower()
        if ext in [".png", ".jpg", ".jpeg"]:
            source = Image.open(source_file)
            source = numpy.array(source)
            target = Image.open(target_file)
            target = numpy.array(target)
        else:
            source = tifffile.imread(source_file)
            target = tifffile.imread(target_file)
        
        source = (source - source.min()) / (source.max() - source.min())
        target = (target - target.min()) / (target.max() - target.min())
        stack = numpy.stack((source, target), dtype=numpy.float32)
        stack = register_images(stack)

        source, target = stack

        dirnames = source_file.split(path)[-1]
        if dirnames.startswith("/"):
            dirnames = dirnames[1:]
        dirnames = os.path.dirname(dirnames)
        basename = os.path.basename(source_file).replace(ext, ".tif")
        
        os.makedirs(os.path.join(output_folder, dirnames), exist_ok=True)
        tifffile.imwrite(
            os.path.join(output_folder, dirnames, basename), source
        )

        dirnames = target_file.split(path)[-1]
        if dirnames.startswith("/"):
            dirnames = dirnames[1:]
        dirnames = os.path.dirname(dirnames)
        basename = os.path.basename(target_file).replace(ext, ".tif")
        
        os.makedirs(os.path.join(output_folder, dirnames), exist_ok=True)
        tifffile.imwrite(
            os.path.join(output_folder, dirnames, basename), target
        )

if __name__ == "__main__":
    main()