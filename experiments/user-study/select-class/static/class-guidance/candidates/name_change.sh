#!/bin/bash 

for file in *MAE_SMALL_STED*; do

    if [ -e "$file" ]; then
        newname="${file/MAE_SMALL_STED/classifier_guidance}"
  
        mv "$file" "$newname"
        echo "Renamed: $file -> $newname"
    fi
done 