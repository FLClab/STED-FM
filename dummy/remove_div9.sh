#!/bin/bash

# Source and destination directories
SRC="/home-local/Frederic/Datasets/ALS/ALS_JM_Fred_unmixed/2023-03-01 (26.2) (dendrites 4c Batch3)"
DEST="/home-local/Frederic/Datasets/ALS/ALS_JM_Fred_unmixed/SynapticDevelopmentDataset"

# Find all files, then filter according to your rules
find "$SRC" -type f \
  ! -path "*/shFUS/*" \
  ! -name "*.tar" \
  -print0 | while IFS= read -r -d '' file; do
    if [[ "$file" == *DIV5* && "$file" == *4DPI* ]]; then
      continue
    fi
    cp "$file" "$DEST"
done