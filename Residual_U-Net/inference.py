import argparse
import os
import torch
import numpy as np
import imageio.v3 as iio
from glob import glob
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_score
from skimage.metrics import peak_signal_noise_ratio as psnr_score


from model import ResidualUNetWithAttention  # update to match your file
from preprocess import build_dataset_from_folder  # from earlier

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on EXR path-traced images")

    parser.add_argument("--test_data_folder", type=str, required=True,
                        help="Folder with test EXR files (must include noisy, normal, albedo)")
    parser.add_argument("--weights_path", type=str, required=True,
                        help="Path to the trained model weights (.pth)")
    parser.add_argument("--output_folder", type=str, default="./output_denoised",
                        help="Where to save denoised images")
    parser.add_argument("--epsilon", type=float, default=1e-2,
                        help="Epsilon to avoid division by zero in exposure normalization")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on")

    return parser.parse_args()

def load_exr(path):
    return iio.imread(path, extension=".exr").astype(np.float32)

def save_exr(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    iio.imwrite(path, data, extension=".exr")

def inference(args):
    device = torch.device(args.device)

    # Load model
    model = ResidualUNetWithAttention(in_channels=9, out_channels=3).to(device)
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model.eval()
    print(f"Loaded model weights from {args.weights_path}")

    # Build dataset from folder
    samples = build_dataset_from_folder(args.test_data_folder)

    total_ssim = 0.0
    total_psnr = 0.0
    num_samples = 0

    for sample in tqdm(samples, desc="Running inference"):
        # Load inputs
        noisy = load_exr(sample['noisy'])
        albedo = load_exr(sample['albedo'])
        normal = load_exr(sample['normal'])
        clean = load_exr(sample['clean']) if 'clean' in sample else None

        inv_albedo = 1.0 / (albedo + args.epsilon)
        model_input = noisy * inv_albedo
        model_input = np.concatenate([model_input, normal, albedo], axis=-1)
        input_tensor = torch.from_numpy(model_input).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            denoised_reflectance = model(input_tensor)[0].cpu().permute(1, 2, 0).numpy()

        denoised = np.clip(denoised_reflectance * albedo, 0.0, 1.0)

        # Save EXR
        filename = os.path.basename(sample['noisy']).replace("_noisy.exr", "_denoised.exr")
        save_path = os.path.join(args.output_folder, filename)
        save_exr(save_path, denoised)

        # If ground truth is available, compute metrics
        if clean is not None:
            clean = np.clip(clean, 0.0, 1.0)
            ssim = ssim_score(clean, denoised, channel_axis=-1, data_range=1.0)
            psnr = psnr_score(clean, denoised, data_range=1.0)
            total_ssim += ssim
            total_psnr += psnr
            num_samples += 1

    if num_samples > 0:
        avg_ssim = total_ssim / num_samples
        avg_psnr = total_psnr / num_samples
        print(f"\nAverage SSIM: {avg_ssim:.4f}")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
    else:
        print("\nSkipped metrics: 'clean' ground truth not found in samples.")

    print(f"Inference complete. Denoised images saved to: {args.output_folder}")



if __name__ == "__main__":
    args = parse_args()
    inference(args)
