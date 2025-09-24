#!/bin/bash

# Chemin du répertoire racine contenant les scènes .pbrt
root_dir="/home/lisic/Documents/Abel/OIDN/oidn_model/training/data/train"

# Parcourir récursivement tous les fichiers dans le répertoire racine
find "$root_dir" -maxdepth 2 -type f -name "*.exr" | while read -r filename; do
    # Obtenir le nom du fichier sans le chemin
    base_filename=$(basename "$filename")
    echo "Rendering $filename"
    python3 "split_exr.py" "$filename"
done

