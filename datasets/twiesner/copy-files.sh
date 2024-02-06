
# Copies the README locally 
rclone copy "valeria-s3:flclab-public/twiesner/Synaptic protein_Paper/SynapticProtein_Paper_Data.xlsx" .
rclone copy "valeria-s3:flclab-public/twiesner/Synaptic protein_Paper/rawdatasource.xlsx" .

# Copies all tifffiles
rclone sync --progressqq \
    "valeria-s3:flclab-public/twiesner/Synaptic protein_Paper" \
    "valeria-s3:flclab-private/FLCDataset/twiesner/synaptic-protein-paper" \
    --filter-from filter-files.txt