import torch
from torch.utils.data import Dataset
import OpenImageIO as oiio
import pyexr
import numpy as np
import os
from glob import glob
import random
from sklearn.model_selection import train_test_split

def create_train_val_datasets(
    data_folder,
    val_split=0.2,
    exposure_match=True,
    transform=None,
    seed=42,
    suffixes=None
):
    """
    Builds EXRDenoiseDataset for training and validation from a folder.
    
    Returns:
        train_dataset, val_dataset
    """
    samples = build_dataset_from_folder(data_folder, suffixes)

    # Shuffle and split
    train_samples, val_samples = train_test_split(
        samples,
        test_size=val_split,
        random_state=seed
    )

    train_dataset = EXRDenoiseDataset(
        train_samples,
        exposure_match=exposure_match,
        transform=transform
    )

    val_dataset = EXRDenoiseDataset(
        val_samples,
        exposure_match=exposure_match,
        transform=None  # Usually no augmentation on val
    )

    print(f"Dataset split: {len(train_samples)} train, {len(val_samples)} val")
    return train_dataset, val_dataset


def build_dataset_from_folder(data_folder, suffixes=None):
    """
    data_folder: path to folder containing EXR files
    suffixes: expected suffixes for each data type (can be customized)
    
    Returns: list of dicts with keys: noisy, clean, normal, albedo
    """
    if suffixes is None:
        suffixes = {
            'noisy': '.hdr.exr',
            #'clean': '_ref.exr',
            'normal': '.nrm.exr',
            'albedo': '.alb.exr',
        }

    # Build index of base filenames (e.g., image_001)
    base_names = set()
    for filepath in glob(os.path.join(data_folder + "/noisies", '*.exr')):
        filename = os.path.basename(filepath)
        for key, suf in suffixes.items():
            if filename.endswith(suf):
                base = filename.replace(suf, '')
                base_names.add(base)
    
    #print(base_names)

    # Build sample list
    samples = []
    for base in sorted(base_names):
        paths = {}
        valid = True
        for key, suf in suffixes.items():
            full_path = os.path.join(data_folder + "/noisies", base + suf)
            if not os.path.exists(full_path):
                print(f"Missing {key} for {base}")
                valid = False
                break
            paths[key] = full_path
        
        clean = os.path.join(data_folder + "/references", base.rsplit("_", 1)[0] + "_ref.hdr.exr")
        if not os.path.exists(full_path):
            print(f"Missing clean for {base}")
            valid = False
            break
        paths["clean"] = clean

        if valid:
            samples.append(paths)

    print(f"Found {len(samples)} valid samples in {data_folder}")
    return samples


def random_crop_flip(input_img, target_img, crop_size=256):
    H, W, _ = input_img.shape
    top = np.random.randint(0, H - crop_size)
    left = np.random.randint(0, W - crop_size)

    input_crop = input_img[top:top+crop_size, left:left+crop_size]
    target_crop = target_img[top:top+crop_size, left:left+crop_size]

    # Random horizontal flip
    if np.random.rand() > 0.5:
        input_crop = np.flip(input_crop, axis=1)
        target_crop = np.flip(target_crop, axis=1)

    return input_crop.copy(), target_crop.copy()


class EXRDenoiseDataset(Dataset):
    def __init__(self, samples, exposure_match=True, transform=None, epsilon=1e-2):
        """
        samples: list of dicts with keys: noisy, clean, normal, albedo
        exposure_match: whether to divide input by albedo
        transform: optional transform (e.g., random crop, flip)
        """
        self.samples = samples
        self.exposure_match = exposure_match
        self.transform = transform
        self.epsilon = epsilon

    def __len__(self):
        return len(self.samples)

    def load_exr(self, path):
        input = oiio.ImageInput.open(path)
        load_num_channels = min(input.spec().nchannels, 3)
        image = input.read_image(subimage=0, miplevel=0, chbegin=0, chend=load_num_channels, format=oiio.FLOAT)
        input.close()
        #img = oiio.ImageInput.open(path) #iio.imread(path, extension=".exr")  # shape: (H, W, 3)
        return image #img.astype(np.float32)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        noisy = self.load_exr(sample['noisy'])     # (H, W, 3)
        clean = self.load_exr(sample['clean'])     # (H, W, 3)
        normal = self.load_exr(sample['normal'])   # (H, W, 3)
        albedo = self.load_exr(sample['albedo'])   # (H, W, 3)

        if noisy.shape != clean.shape:
            clean = np.resize(clean, noisy.shape)
            clean = np.nan_to_num(clean)
        
        if noisy.shape != albedo.shape:
            albedo = np.resize(albedo, noisy.shape)
            albedo = np.nan_to_num(albedo)
        
        if noisy.shape != normal.shape:
            normal = np.resize(normal, noisy.shape)
            normal = np.nan_to_num(normal)
        
        if self.exposure_match:
            # Avoid division by zero with epsilon
            inv_albedo = 1.0 / (albedo + self.epsilon)
            noisy_input = noisy * inv_albedo
            clean_target = clean * inv_albedo
        else:
            noisy_input = noisy
            clean_target = clean

        # Stack input channels: noisy RGB + normal + albedo → shape: (H, W, 9)
        input_concat = np.concatenate([noisy_input, normal, albedo], axis=-1)

        if self.transform:
            input_concat, clean_target = self.transform(input_concat, clean_target)

        # Convert to tensor and permute to (C, H, W)
        input_tensor = torch.from_numpy(input_concat).permute(2, 0, 1)
        target_tensor = torch.from_numpy(clean_target).permute(2, 0, 1)

        return input_tensor, target_tensor
