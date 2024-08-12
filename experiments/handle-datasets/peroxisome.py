
import os, glob
import numpy 
import tifffile 
import random

from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, "..")

from DEFAULTS import BASE_PATH

def main():

    files = sorted(glob.glob(os.path.join(BASE_PATH, "evaluation-data", "peroxisome", "**/*.tif*"), recursive=True))

    # Filter files from the dataset
    files = filter(lambda x: "FigureImages" not in x, files)
    files = filter(lambda x: "_conf" not in x, files)
    files = list(files)

    for file in files:
        try:
            tiff = tifffile.imread(file)
        except Exception as e:
            print(f"Error reading file {file}")
            print(e)
            files.remove(file)

    classes = set()
    for file in files:
        classes.add(file.split("/")[-2].split("_")[0])
    print(list(sorted(classes)))

    training_files = list(filter(lambda x: "Triplo" not in x, files))
    training_files, validation_files = train_test_split(training_files, test_size=0.2, random_state=42)
    testing_files = list(filter(lambda x: "Triplo" in x, files))

    print("Training files: ", len(training_files))
    print("Validation files: ", len(validation_files))
    print("Testing files: ", len(testing_files))

    for files in [training_files, validation_files, testing_files]:
        classes = set()
        for file in files:
            classes.add(file.split("/")[-2].split("_")[0])
        print(list(sorted(classes)))

    # Save the files
    with open(os.path.join(BASE_PATH, "evaluation-data", "peroxisome", "peroxisome-training.txt"), "w") as f:
        for file in training_files:
            file = file.replace(BASE_PATH, "")
            f.write(file + "\n")
    with open(os.path.join(BASE_PATH, "evaluation-data", "peroxisome", "peroxisome-validation.txt"), "w") as f:
        for file in validation_files:
            file = file.replace(BASE_PATH, "")
            f.write(file + "\n")
    with open(os.path.join(BASE_PATH, "evaluation-data", "peroxisome", "peroxisome-testing.txt"), "w") as f:
        for file in testing_files:
            file = file.replace(BASE_PATH, "")
            f.write(file + "\n")

if __name__ == "__main__":
    main()