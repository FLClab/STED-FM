
# Copies the README locally 
rclone copy valeria-s3:flclab-public/oferguson/Inserts_Images_and_Masks/README.txt .

# Copies all tifffiles
rclone sync --progress \
    valeria-s3:flclab-public/oferguson/Inserts_Images_and_Masks \
    valeria-s3:flclab-private/FLCDataset/oferguson/Inserts_Images_and_Masks \
    --filter-from filter-files.txt