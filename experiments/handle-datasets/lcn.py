
import os, glob
from matplotlib import image
import numpy
import tifffile

from sklearn.model_selection import train_test_split
from stedfm.DEFAULTS import BASE_PATH

SKIPFRAMES = 10

def export_to_tiff(stack, mask_stack, path, filename):
    # Ensure directory exists
    os.makedirs(path, exist_ok=True)

    for i in range(0, stack.shape[0], SKIPFRAMES):
        frame = stack[i]
        mask = mask_stack[i]
        m, M = frame.min(), frame.max()
        frame = (frame - m) / (M - m + 1e-8)

        tifffile.imwrite(os.path.join(path, filename.replace("_img.tif", f"_frame{i:03d}.tif")), frame.astype(numpy.float32))
        tifffile.imwrite(os.path.join(path, filename.replace("_img.tif", f"_frame{i:03d}_mask.tif")), mask.astype(numpy.float32))

def main():

    for split in ["training", "validation", "testing"]:
        dir_path = os.path.join(BASE_PATH, "segmentation-data", "lcn", split)
        if os.path.exists(dir_path):
            print(f"Removing existing directory {dir_path}...")
            import shutil
            shutil.rmtree(dir_path)

    image_files = sorted(glob.glob(os.path.join(BASE_PATH, "segmentation-data", "lcn", "**/*.tif*"), recursive=True))
    image_files = list(filter(lambda x: "raw" in x, image_files))
    image_files = list(filter(lambda x: "_seg" not in x, image_files))
    
    training_files, validation_files = train_test_split(image_files, test_size=0.3, random_state=42)
    validation_files, testing_files = train_test_split(validation_files, test_size=0.5, random_state=42)

    print("Training files: ", len(training_files))
    print("Validation files: ", len(validation_files))
    print("Testing files: ", len(testing_files))

    for file_list, split in zip([training_files, validation_files, testing_files], ["training", "validation", "testing"]):
        for file in file_list:
            try:
                stack = tifffile.imread(file)
                mask_stack = tifffile.imread(file.replace("_img.tif", "_seg.tif"))
            except Exception as e:
                print(f"Error reading file {file}")
                print(e)
                continue
            export_to_tiff(stack, mask_stack, os.path.join(BASE_PATH, "segmentation-data", "lcn", split), os.path.basename(file))

if __name__ == "__main__":
    main()