
import os, glob
import tifffile
import numpy

from stedfm.DEFAULTS import BASE_PATH

def group_files_by_image(files):
    grouped_files = {}
    for f in files:
        basename = os.path.basename(f)
        image_id = basename.split(" ")[:2]
        image_id = " ".join(image_id)
        if image_id not in grouped_files:
            grouped_files[image_id] = []
        grouped_files[image_id].append(f)
    for image_id in grouped_files:
        grouped_files[image_id] = list(sorted(grouped_files[image_id]))
    return grouped_files

def make_dataset(dataset_path: str, outdir: str = f"{BASE_PATH}/evaluation-data/reg3b/processed"):
    
    files = glob.glob(os.path.join(dataset_path, "**/*.tif"), recursive=True)
    files = list(filter(lambda f: "After translation and cropping" in f, files))
    files = list(filter(lambda f: "confocal" not in f, files))

    training_files = [f for f in files if "WT1" in f or "WT2" in f or "OE1" in f or "OE2" in f]
    testing_files = [f for f in files if "WT3" in f or "OE3" in f]

    for fold_name, files in zip(["train", "test"], [training_files, testing_files]):
        print(f"Processing {fold_name} files...")

        groups = group_files_by_image(files)
        
        for image_id, files in groups.items():
            stack = []
            for f in files:
                image = tifffile.imread(f)

                with tifffile.TiffFile(f) as tif:
                    resolution = tif.pages[0].tags['XResolution'].value
                    resolution = resolution[0] / resolution[1]
                stack.append(image)
            stack = numpy.stack(stack, axis=0)

            dirname = os.path.dirname(f)
            directories = dirname.split(os.sep)
            condition = directories[-3]
            dst = os.path.join(outdir, fold_name, condition)
            os.makedirs(dst, exist_ok=True)

            tifffile.imwrite(
                os.path.join(dst, image_id + ".tiff"),
                stack.astype(numpy.uint16),
                resolution=(resolution, resolution),
                imagej=True,
                metadata={"unit" : "um", "mode" : "composite"}
            )
def main():
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default=f"{BASE_PATH}/evaluation-data/reg3b/Reg3b FINAL")
    parser.add_argument("--export-to-tiff", action="store_true")
    args = parser.parse_args()   

    make_dataset(args.dataset_path)

if __name__=="__main__":
    main()