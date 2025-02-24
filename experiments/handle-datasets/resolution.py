
import os, glob
import numpy 
import tifffile 
import random
import h5py

from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm.auto import tqdm
from banditopt.objectives import Resolution 

import sys
sys.path.insert(0, "..")

from DEFAULTS import BASE_PATH

CHANNEL = 1
SIZE = 224

random.seed(42)
numpy.random.seed(42)

def main():

    for dataset_name in ["training", "validation", "testing"]:
        with h5py.File(os.path.join(BASE_PATH, "evaluation-data", "resolution-dataset", f"{dataset_name}.hdf5"), "w") as f:
            # overwrite the file
            pass

    files = sorted(glob.glob(os.path.join(BASE_PATH, "evaluation-data", "low-high-quality", "**/*.tif*"), recursive=True))

    training_files, validation_files = train_test_split(files, test_size=0.3, random_state=42)
    validation_files, testing_files = train_test_split(validation_files, test_size=0.5, random_state=42)

    print("Training files: ", len(training_files))
    print("Validation files: ", len(validation_files))
    print("Testing files: ", len(testing_files))

    for dataset_name, files in [("training", training_files), ("validation", validation_files), ("testing", testing_files)]:

        with h5py.File(os.path.join(BASE_PATH, "evaluation-data", "resolution-dataset", f"{dataset_name}.hdf5"), "a") as f:

            X_dset = f.create_dataset(
                "data", shape=(0, SIZE, SIZE), 
                maxshape=(None, 224, 224), 
                chunks=True, dtype=numpy.float32)

            y_dset = f.create_dataset(
                "labels", shape=(0,), 
                maxshape=(None,), 
                chunks=True, dtype=numpy.float32)

            for file in tqdm(files):
                crops, labels = [], []
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
                            resolution_objective = Resolution(pixelsize=20e-9)
                            resolution = resolution_objective.evaluate([crop], None, None, None, None)

                            if resolution == resolution_objective.res_cap:
                                # Calculation of the resolution did not converge, skipping this crop
                                continue
                            labels.append(resolution)
                            crops.append(crop)

                except Exception as e:
                    print(f"Error reading file {file}")
                    print(e)
            
                X_dset.resize(X_dset.shape[0] + len(crops), axis=0)
                X_dset[-len(crops):] = crops

                y_dset.resize(y_dset.shape[0] + len(labels), axis=0)
                y_dset[-len(labels):] = labels

if __name__ == "__main__":

    main()