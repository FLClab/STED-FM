import os
import glob
import tifffile
import numpy

from skimage.transform import resize

def export_single_frame(filename: str, stack: numpy.ndarray):
    for i, frame in enumerate(stack):
        tifffile.imwrite(f"{filename}_{i:04}.tif", frame)

def main():

    path = "/home-local2/projects/SSL/denoising-data/lioness-lqhq"
    for split in ["training data", "testing data"]:
        low_paths = sorted(glob.glob(os.path.join(path, split, "low", "*.tif")))
        gt_paths = sorted(glob.glob(os.path.join(path, split, "gt", "*.tif")))

        assert len(low_paths) == len(gt_paths)

        for low_path in low_paths:
            stack = tifffile.imread(low_path)
            if stack.shape[-2] < 256 or stack.shape[-1] < 256:
                stack = resize(stack, (stack.shape[0], 256, 256), anti_aliasing=False, preserve_range=True, order=0)

            filename = os.path.basename(low_path)
            filename = os.path.splitext(filename)[0]
            os.makedirs(os.path.join(path, split, "low_single-frames"), exist_ok=True)
            export_single_frame(os.path.join(path, split, "low_single-frames", filename), stack)

        for gt_path in gt_paths:
            stack = tifffile.imread(gt_path)
            if stack.shape[-2] < 256 or stack.shape[-1] < 256:
                stack = resize(stack, (stack.shape[0], 256, 256), anti_aliasing=False, preserve_range=True, order=0)

            filename = os.path.basename(gt_path)
            filename = os.path.splitext(filename)[0]
            os.makedirs(os.path.join(path, split, "gt_single-frames"), exist_ok=True)
            export_single_frame(os.path.join(path, split, "gt_single-frames", filename), stack)

if __name__ == "__main__":
    main()