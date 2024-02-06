
# Copies the README locally 
rclone copy valeria-s3:flclab-private/ml-datasets/factin-dendrite-lavoiecardinal/readme.txt .

# Copies all tifffiles
rclone sync --progress --dry-run \
    valeria-s3:flclab-private/ml-datasets/factin-dendrite-lavoiecardinal \
    valeria-s3:flclab-private/FLCDataset/flavoiecardinal/factin-dendrite-lavoiecardinal \
    --filter-from filter-files.txt