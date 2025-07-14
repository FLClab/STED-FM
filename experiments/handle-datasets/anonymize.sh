#!/bin/bash

dir="/home-local/Frederic/Datasets/ALS/ALS_JM_Fred_unmixed/SynapticDevelopmentDataset"

index=0
for filepath in "$dir"/*; do
    if [ -f "$filepath" ]; then
        ext="${filepath##*.}"
        mv "$filepath" "$dir/img_${index}.${ext}"
        index=$((index + 1))
    fi
done
