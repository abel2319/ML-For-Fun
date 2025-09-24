#!/bin/bash

root_dir="/home/lisic/Documents/Abel/OIDN/oidn_model/training/data/valid"
# Parcourir tous les sous-répertoires
find "$root_dir" -maxdepth 1 -type d | while read -r dir; do
    if [ "$dir" != "." ]; then
        echo "Processing directory: $dir"
        python3 "resize_images.py" "$dir"
    fi
done

echo "All images have been resized and originals moved to 'old' directories."