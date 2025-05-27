import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def train(
    model,
    train_dataset,
    val_dataset=None,
    num_epochs=50,
    batch_size=8,
    learning_rate=1e-4,
    device='cuda',
    save_path='checkpoints',
    save_every=10,
):
    os.makedirs(save_path, exist_ok=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1) if val_dataset else None

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.L1Loss()

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for inputs, targets in progress:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1} average training loss: {avg_train_loss:.6f}")

        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_inputs = val_inputs.to(device)
                    val_targets = val_targets.to(device)

                    val_outputs = model(val_inputs)
                    val_loss += criterion(val_outputs, val_targets).item()
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(f"Epoch {epoch+1} validation loss: {avg_val_loss:.6f}")

        # Save model checkpoint
        if (epoch + 1) % save_every == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = os.path.join(save_path, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}: {checkpoint_path}")

    # Plot losses after training
    plt.figure(figsize=(10,6))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    if val_dataset:
        plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(save_path, f'loss_curve_{epoch + 1}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss curve saved to {plot_path}")
