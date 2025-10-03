

import os, glob
import numpy 
import tifffile 
import random

from sklearn.model_selection import train_test_split

from stedfm.DEFAULTS import BASE_PATH

def main():

    training_files, validation_files, testing_files = [], [], []
    for folder in ["FCCP/DMSO", "FCCP/FCCP"]:
        files = list(sorted(glob.glob(os.path.join(BASE_PATH, "evaluation-data", "BBBC053", folder, "**/*.tif*"), recursive=True)))

        # for file in files:
        #     try:
        #         tiff = tifffile.imread(file)
        #     except Exception as e:
        #         print(f"Error reading file {file}")
        #         print(e)
        #         files.remove(file)

        training, validation = train_test_split(files, test_size=0.3, random_state=42)
        validation, testing = train_test_split(validation, test_size=0.5, random_state=42)

        training_files.extend(training)
        validation_files.extend(validation)
        testing_files.extend(testing)

    print("Training files: ", len(training_files))
    print("Validation files: ", len(validation_files))
    print("Testing files: ", len(testing_files))

    with open(os.path.join(BASE_PATH, "evaluation-data", "BBBC053", "BBBC053-training.txt"), "w") as f:
        for file in training_files:
            file = file.replace(BASE_PATH, "")
            if file.startswith("/"):
                file = file[1:]
            f.write(file + "\n")
    with open(os.path.join(BASE_PATH, "evaluation-data", "BBBC053", "BBBC053-validation.txt"), "w") as f:
        for file in validation_files:
            file = file.replace(BASE_PATH, "")
            if file.startswith("/"):
                file = file[1:]
            f.write(file + "\n")
    with open(os.path.join(BASE_PATH, "evaluation-data", "BBBC053", "BBBC053-testing.txt"), "w") as f:
        for file in testing_files:
            file = file.replace(BASE_PATH, "")
            if file.startswith("/"):
                file = file[1:]
            f.write(file + "\n")

if __name__ == "__main__":

    main()