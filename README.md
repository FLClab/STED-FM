# flc-dataset
Dataset management of FLClab organization

## Datasets

In this folder, we include datasets from different users of the lab. A README-type file is provided keep track of the images. A `copy-files` is included to launch the copy-paste of all images.

## Steps update datasets

1. Copy data from users in their respective folder
    1. You can use `--dry-run` option for matching
1. Update the `metadata.py` file to match all possible images
1. Sync dataset from Valeria to computer
1. Run `dataset.py` to update the tar file
