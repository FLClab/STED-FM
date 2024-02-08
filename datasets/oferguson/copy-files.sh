
# Copies the README locally 
rclone copyto valeria-s3:flclab-public/oferguson/Inserts_Images_and_Masks/README.txt ./readme-inserts.txt
rclone copyto valeria-s3:flclab-public/oferguson/Manual_Mitos/README.txt ./readme-manual-mitos.txt

# Copies all tifffiles in Inserts
rclone copy --progress \
    valeria-s3:flclab-public/oferguson/Inserts_Images_and_Masks \
    valeria-s3:flclab-private/FLCDataset/oferguson/Inserts_Images_and_Masks \
    --filter-from filter-files-inserts.txt

# Copies all tifffiles in Manual Mitos
rclone copy --progress \
    valeria-s3:flclab-public/oferguson/Manual_Mitos \
    valeria-s3:flclab-private/FLCDataset/oferguson/Manual_Mitos \
    --filter-from filter-files-manual-mitos.txt    