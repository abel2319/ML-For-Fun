import os
import glob
import OpenImageIO as oiio
from shutil import copy2

def get_image_size(filepath):
    img = oiio.ImageInput.open(filepath)
    if not img:
        print(f"Could not open {filepath}")
        return None
    spec = img.spec()
    width, height = spec.width, spec.height
    img.close()
    return (width, height)

def resize_image(input_path, output_path, target_size):
    cmd = f"oiiotool {input_path} --resize {target_size[0]}x{target_size[1]} -o {output_path}"
    os.system(cmd)

def process_directory(directory):
    # Trouver toutes les images pertinentes
    images_128 = glob.glob(os.path.join(directory, "*_128spp.hdr.exr"))
    images_ref = glob.glob(os.path.join(directory, "*_ref.hdr.exr"))
    all_images = images_128 + images_ref
    
    if not all_images:
        return
    
    # Trouver la plus petite taille parmi toutes les images
    min_size = None
    for img in all_images:
        size = get_image_size(img)
        if size:
            if min_size is None or (size[0] * size[1]) < (min_size[0] * min_size[1]):
                min_size = size
    
    if not min_size:
        print(f"No valid images found in {directory}")
        return
    
    # Créer le dossier old s'il n'existe pas
    old_dir = os.path.join(directory, "old")
    os.makedirs(old_dir, exist_ok=True)
    
    # Redimensionner toutes les images
    for img in all_images:
        # Copier l'original dans old
        img_name = os.path.basename(img)
        old_path = os.path.join(old_dir, img_name)
        copy2(img, old_path)
        
        # Redimensionner l'image
        resize_image(img, img, min_size)
        print(f"Resized {img} to {min_size[0]}x{min_size[1]}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python resize_images.py <directory>")
        sys.exit(1)
    
    process_directory(sys.argv[1])