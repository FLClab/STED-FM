
import os, glob
import re
import numpy 
import tifffile 
import random

from sklearn.model_selection import train_test_split

from stedfm.DEFAULTS import BASE_PATH

def match_condition(files, condition):
    pattern = f"_[A-Z]{condition}_"
    matched_files = [f for f in files if re.search(pattern, os.path.basename(f))]
    return matched_files

def main():

    training_files, validation_files, testing_files = [], [], []
    files = list(sorted(glob.glob(os.path.join(BASE_PATH, "evaluation-data", "BBBC026", "**/*.png*"), recursive=True)))

    for condition in ["01", "23"]:
        matched_files = match_condition(files, condition)
        training, validation = train_test_split(matched_files, test_size=0.3, random_state=42)
        validation, testing = train_test_split(validation, test_size=0.5, random_state=42)

    # for file in files:
    #     try:
    #         tiff = tifffile.imread(file)
    #     except Exception as e:
    #         print(f"Error reading file {file}")
    #         print(e)
    #         files.remove(file)

        training_files.extend(training)
        validation_files.extend(validation)
        testing_files.extend(testing)

    print("Training files: ", len(training_files))
    print("Validation files: ", len(validation_files))
    print("Testing files: ", len(testing_files))

    with open(os.path.join(BASE_PATH, "evaluation-data", "BBBC026", "BBBC026-training.txt"), "w") as f:
        for file in training_files:
            file = file.replace(BASE_PATH, "")
            if file.startswith("/"):
                file = file[1:]
            f.write(file + "\n")
    with open(os.path.join(BASE_PATH, "evaluation-data", "BBBC026", "BBBC026-validation.txt"), "w") as f:
        for file in validation_files:
            file = file.replace(BASE_PATH, "")
            if file.startswith("/"):
                file = file[1:]
            f.write(file + "\n")
    with open(os.path.join(BASE_PATH, "evaluation-data", "BBBC026", "BBBC026-testing.txt"), "w") as f:
        for file in testing_files:
            file = file.replace(BASE_PATH, "")
            if file.startswith("/"):
                file = file[1:]
            f.write(file + "\n")

if __name__ == "__main__":

    main()