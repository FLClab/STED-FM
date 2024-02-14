
# Copies the README locally 
rclone copyto "valeria-s3:flclab-archive/braulier/Flavie/TIFF files for Flavie/2-Color STED CaMKII-ACTIN/README.txt" ./readme.txt

# Copies all tifffiles in Inserts
rclone copy --progress --update --dry-run \
    "valeria-s3:flclab-archive/braulier/Flavie/TIFF files for Flavie/2-Color STED CaMKII-ACTIN" \
    "valeria-s3:flclab-private/FLCDataset/braulier/2-Color STED CaMKII-ACTIN" \
    --filter-from filter-files.txt
