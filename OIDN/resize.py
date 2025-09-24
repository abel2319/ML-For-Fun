import os
import OpenImageIO as oiio

def resize_images_in_directory(root_dir):
    for subdir, _, files in os.walk(root_dir):
        # Filter files to only include the relevant image pairs
        image_files = [f for f in files if f.endswith('_128spp.hdr.exr') or f.endswith('_ref.hdr.exr')]

        if len(image_files) != 2:
            continue  # Skip directories that don't have exactly two relevant images

        min_width, min_height = float('inf'), float('inf')

        # Step 1: Find the minimum dimensions for the pair
        for file in image_files:
            file_path = os.path.join(subdir, file)
            img = oiio.ImageInput.open(file_path)
            spec = img.spec()
            width, height = spec.width, spec.height
            if width < min_width or height < min_height:
                min_width, min_height = width, height
            img.close()

        old_dir = os.path.join(subdir, 'old')
        os.makedirs(old_dir, exist_ok=True)

        # Step 2: Resize images and save originals
        for file in image_files:
            file_path = os.path.join(subdir, file)
            img = oiio.ImageInput.open(file_path)
            spec = img.spec()
            pixels = oiio.ImageBuf(spec)
            img.read_image(oiio.FLOAT, pixels)
            img.close()

            # Save original to 'old' directory
            old_file_path = os.path.join(old_dir, file)
            output = oiio.ImageOutput.create(old_file_path)
            output.write_image(oiio.FLOAT, pixels)
            output.close()

            # Resize image
            resized_pixels = oiio.ImageBuf(oiio.ImageSpec(min_width, min_height, spec.nchannels, spec.format))
            oiio.ImageBufAlgo.resize(resized_pixels, pixels)

            # Save resized image
            output = oiio.ImageOutput.create(file_path)
            output.write_image(oiio.FLOAT, resized_pixels)
            output.close()

# Example usage
resize_images_in_directory('/home/lisic/Documents/Abel/OIDN/oidn_model/training/data/train')
