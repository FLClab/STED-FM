
import os, glob
import numpy 
import tifffile 
import random

from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, "..")

from DEFAULTS import BASE_PATH

def main():

    files = sorted(glob.glob(os.path.join(BASE_PATH, "evaluation-data", "DL-SIM", "**/*.tif*"), recursive=True))
    files = list(filter(lambda x: "testing" not in x, files))

    for file in files:
        try:
            tiff = tifffile.imread(file)
        except Exception as e:
            print(f"Error reading file {file}")
            print(e)
            files.remove(file)

    training_files, validation_files = train_test_split(files, test_size=0.3, random_state=42)

    testing_files = sorted(glob.glob(os.path.join(BASE_PATH, "evaluation-data", "DL-SIM", "**/*.tif*"), recursive=True))
    testing_files = list(filter(lambda x: "testing" in x, testing_files))
    for file in testing_files:
        try:
            tiff = tifffile.imread(file)
        except Exception as e:
            print(f"Error reading file {file}")
            print(e)
            testing_files.remove(file)

    print("Training files: ", len(training_files))
    print("Validation files: ", len(validation_files))
    print("Testing files: ", len(testing_files))

    with open(os.path.join(BASE_PATH, "evaluation-data", "DL-SIM", "DL-SIM-training.txt"), "w") as f:
        for file in training_files:
            file = file.replace(BASE_PATH, "")
            f.write(file + "\n")
    with open(os.path.join(BASE_PATH, "evaluation-data", "DL-SIM", "DL-SIM-validation.txt"), "w") as f:
        for file in validation_files:
            file = file.replace(BASE_PATH, "")
            f.write(file + "\n")
    with open(os.path.join(BASE_PATH, "evaluation-data", "DL-SIM", "DL-SIM-testing.txt"), "w") as f:
        for file in testing_files:
            file = file.replace(BASE_PATH, "")
            f.write(file + "\n")

if __name__ == "__main__":

    main()