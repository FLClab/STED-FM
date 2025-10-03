############################################################
# Download BBBC052 dataset from the Broad Bioimage Benchmark Collection
# https://bbbc.broadinstitute.org/BBBC052
############################################################

# links=(
#     https://data.broadinstitute.org/bbbc/BBBC052/CK-666-20220127T152118Z-001.zip
#     https://data.broadinstitute.org/bbbc/BBBC052/Cofilin1KD-20220127T152122Z-001.zip
#     https://data.broadinstitute.org/bbbc/BBBC052/FP4-Mito-20220127T152125Z-001.zip
#     https://data.broadinstitute.org/bbbc/BBBC052/PFN1KO-20220127T152128Z-001.zip
#     https://data.broadinstitute.org/bbbc/BBBC052/TBeta4KD-20220127T152130Z-001.zip
# )
# names=(
#     CK-666
#     Cofilin1KD
#     FP4-Mito
#     PFN1KO
#     TBeta4KD
# )
# outpath="/home-local2/projects/SSL/evaluation-data/BBBC052/"
# mkdir -p $outpath
# for i in "${!links[@]}"; do
#     wget -O "${outpath}${names[i]}.zip" "${links[i]}"
#     unzip -o "${outpath}${names[i]}.zip" -d "${outpath}${names[i]}"
#     rm "${outpath}${names[i]}.zip"
# done
# current_dir=$(pwd)
# cd $outpath
# # Move files from subdirectories to main directory and remove empty subdirectories
# mv CK-666/CK-666/* CK-666/
# mv Cofilin1KD/Cofilin1KD/* Cofilin1KD/
# mv FP4-Mito/FP4-Mito/* FP4-Mito/
# mv PFN1KO/PFN1KO/* PFN1KO/
# mv TBeta4KD/TBeta4KD/* TBeta4KD/
# rmdir CK-666/CK-666/ Cofilin1KD/Cofilin1KD/ FP4-Mito/FP4-Mito/ PFN1KO/PFN1KO/ TBeta4KD/TBeta4KD/
# echo "Downloaded and extracted all datasets."
# cd $current_dir

############################################################
# Download BBBC053 dataset from the Broad Bioimage Benchmark Collection
# https://bbbc.broadinstitute.org/BBBC053
############################################################

# outpath="/home-local2/projects/SSL/evaluation-data/BBBC053/"
# mkdir -p $outpath
# wget -O "${outpath}BBBC053.zip" "https://data.broadinstitute.org/bbbc/BBBC053/FCCP-20220127T153841Z-001.zip"
# unzip -o "${outpath}BBBC053.zip" -d "${outpath}"
# rm "${outpath}BBBC053.zip"

############################################################
# Download BBBC026 dataset from the Broad Bioimage Benchmark Collection
# https://bbbc.broadinstitute.org/BBBC026
############################################################

outpath="/home-local2/projects/SSL/evaluation-data/BBBC026/"
mkdir -p $outpath
wget -O "${outpath}BBBC026.zip" "https://data.broadinstitute.org/bbbc/BBBC026/BBBC026_v1_images.zip"
unzip -o "${outpath}BBBC026.zip" -d "${outpath}"
rm "${outpath}BBBC026.zip"