import argparse
import os
import torch
from torch.utils.data import DataLoader
from preprocess import create_train_val_datasets, random_crop_flip  # import your dataset utilities
from model import ResidualUNetWithAttention  # import your model
from train import train  # import your train function

def parse_args():
    parser = argparse.ArgumentParser(description="Train Residual U-Net Denoiser")

    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root folder containing the EXR dataset")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints and models")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of dataset to use for validation")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for optimizer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on (cuda or cpu)")

    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Using device: {args.device}")

    # Create datasets
    train_dataset, val_dataset = create_train_val_datasets(
        data_folder=args.data_root,
        val_split=args.val_split,
        exposure_match=True,
        transform=random_crop_flip
    )

    # DataLoaders
    #train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    #val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize model
    model = ResidualUNetWithAttention(in_channels=9, out_channels=3)
    model.to(args.device)

    # Train
    train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        save_path=args.save_dir,
        save_every=10,
    )

if __name__ == "__main__":
    main()
