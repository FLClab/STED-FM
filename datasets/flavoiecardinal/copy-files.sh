
# Copies the README locally 
rclone copyto valeria-s3:flclab-private/ml-datasets/factin-dendrite-lavoiecardinal/readme.txt ./readme-actin.txt
rclone copyto valeria-s3:flclab-private/ml-datasets/optim-lavoiecardinal/readme.txt ./readme-optim.txt

# Copies all tifffiles from Actin-Paper
rclone sync --progress \
    valeria-s3:flclab-private/ml-datasets/factin-dendrite-lavoiecardinal \
    valeria-s3:flclab-private/FLCDataset/flavoiecardinal/factin-dendrite-lavoiecardinal \
    --filter-from filter-files-actin.txt

# Copies files from Optim-Paper
rclone sync --progress \
    valeria-s3:flclab-private/ml-datasets/optim-lavoiecardinal \
    valeria-s3:flclab-private/FLCDataset/flavoiecardinal/optim-lavoiecardinal \
    --filter-from filter-files-optim.txt