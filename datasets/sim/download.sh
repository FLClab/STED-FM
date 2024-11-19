
DIR="/home-local2/projects/SSL/ssl-data/sim"

# BioSR data download
https://figshare.com/ndownloader/files/25714514
https://figshare.com/ndownloader/files/25714583
https://figshare.com/ndownloader/files/25714658
https://figshare.com/ndownloader/files/25714772
https://figshare.com/ndownloader/files/25944599

# 3D RCAN data download
if [ ! -f "$DIR/Denoising1.zip" ]; then
    wget -O "$DIR/Denoising1.zip" "https://zenodo.org/record/4651921/files/Denoising1.zip?download=1"
fi
if [ ! -f "$DIR/Expansion_Microscopy.zip" ]; then
    wget -O "$DIR/Expansion_Microscopy.zip" "https://zenodo.org/record/4651921/files/Expansion_Microscopy.zip?download=1"
fi
if [ ! -f "$DIR/live_cell_test_data.zip" ]; then
    wget -O "$DIR/live_cell_test_data.zip" "https://zenodo.org/records/4651921/files/live_cell_test_data.zip?download=1"
fi

# DeepBACS data download
if [ ! -f "$DIR/DeepBacs_Data_Super-resolution_prediction_S.aureus.zip" ]; then
    wget -O "$DIR/DeepBacs_Data_Super-resolution_prediction_S.aureus.zip" "https://zenodo.org/records/5551141/files/DeepBacs_Data_Super-resolution_prediction_S.aureus.zip?download=1"
fi
if [ ! -f "$DIR/DeepBacs_Data_Super-resolution_prediction_E.coli.zip" ]; then
    wget -O "$DIR/DeepBacs_Data_Super-resolution_prediction_E.coli.zip" "https://zenodo.org/records/5551153/files/DeepBacs_Data_Super-resolution_prediction_E.coli.zip?download=1"
fi

# EMTB data download
if [ ! -f "$DIR/DeepLearning_test_dataset(EMTB).zip" ]; then
    wget -O "$DIR/DeepLearning_test_dataset(EMTB).zip" "https://zenodo.org/records/6727773/files/DeepLearning_test_dataset(EMTB).zip?download=1"
fi
