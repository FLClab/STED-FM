
import os, glob
import numpy 
import tifffile 
import random
import h5py

from sklearn.model_selection import train_test_split
from collections import defaultdict

import sys
sys.path.insert(0, "..")

from DEFAULTS import BASE_PATH

CHANNEL = 1
SIZE = 224

random.seed(42)
numpy.random.seed(42)

def main():

    for dataset_name in ["training", "validation", "testing"]:
        with h5py.File(os.path.join(BASE_PATH, "evaluation-data", "low-high-quality", f"{dataset_name}.hdf5"), "w") as f:
            # overwrite the file
            pass

    files = sorted(glob.glob(os.path.join(BASE_PATH, "evaluation-data", "low-high-quality", "**/*.tif*"), recursive=True))

    lq_files_per_condition = defaultdict(list)
    hq_files_per_condition = defaultdict(list)
    for file in files:
        condition = file.split(os.path.sep)[-2]

        if "FarRedDyes" in file:
            LOW_QUALITY = "_5percent"
            HIGH_QUALITY = "_15percent"
        else:
            LOW_QUALITY = "_10percent"
            HIGH_QUALITY = "_30percent"

        if LOW_QUALITY in file:
            lq_files_per_condition[condition].append(file)
        elif HIGH_QUALITY in file:
            hq_files_per_condition[condition].append(file)
        else:
            pass
        
    for quality, files_per_condition in [("low", lq_files_per_condition), ("high", hq_files_per_condition)]:
        for condition, files in files_per_condition.items():
            print(f"Condition {condition}: {len(files)} files")

            training_files, validation_files = train_test_split(files, test_size=0.3, random_state=42)
            validation_files, testing_files = train_test_split(validation_files, test_size=0.5, random_state=42)

            print("Training files: ", len(training_files))
            print("Validation files: ", len(validation_files))
            print("Testing files: ", len(testing_files))

            for dataset_name, files in [("training", training_files), ("validation", validation_files), ("testing", testing_files)]:

                with h5py.File(os.path.join(BASE_PATH, "evaluation-data", "low-high-quality", f"{dataset_name}.hdf5"), "a") as f:
                    if not quality in f:
                        f.create_group(quality)
                    quality_group = f[quality]

                    if not condition in quality_group:
                        dset = quality_group.create_dataset(
                            condition, shape=(0, SIZE, SIZE), 
                            maxshape=(None, 224, 224), 
                            chunks=True, dtype=numpy.float32)
                    dset = quality_group[condition]

                    crops = []
                    for file in files:
                        try:
                            data = tifffile.imread(file)[:, CHANNEL]
                            data = numpy.sum(data[10: 110], axis=0)
                            m, M = data.min(), data.max()
                            data = (data - m) / (M - m)

                            if data.shape[0] < SIZE or data.shape[1] < SIZE:
                                continue
                            for j in range(0, data.shape[0]-SIZE, SIZE):
                                for i in range(0, data.shape[1]-SIZE, SIZE):
                                    crop = data[j:j+SIZE, i:i+SIZE]
                                    crops.append(crop)
                        except Exception as e:
                            print(f"Error reading file {file}")
                            print(e)
                    
                    dset.resize(dset.shape[0] + len(crops), axis=0)
                    dset[-len(crops):] = crops

    # with open(os.path.join(BASE_PATH, "evaluation-data", "DL-SIM", "DL-SIM-training.txt"), "w") as f:
    #     for file in training_files:
    #         file = file.replace(BASE_PATH, "")
    #         f.write(file + "\n")
    # with open(os.path.join(BASE_PATH, "evaluation-data", "DL-SIM", "DL-SIM-validation.txt"), "w") as f:
    #     for file in validation_files:
    #         file = file.replace(BASE_PATH, "")
    #         f.write(file + "\n")
    # with open(os.path.join(BASE_PATH, "evaluation-data", "DL-SIM", "DL-SIM-testing.txt"), "w") as f:
    #     for file in testing_files:
    #         file = file.replace(BASE_PATH, "")
    #         f.write(file + "\n")

if __name__ == "__main__":

    main()