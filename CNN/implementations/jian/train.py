import os
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import pandas as pd
from CNN import CNN
from datasets import train_dataset, val_dataset, test_dataset

from pytorch_utils import set_seed, get_device, get_dataloader, save_checkpoint


# -----------------------------
# 0. Training / Evaluation Functions
# -----------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


if __name__ == '__main__':

    # -----------------------------
    # 1. Reproducibility & Device
    # -----------------------------

    SEED = 1
    set_seed(SEED, deterministic=True)
    device = get_device()
    print(f"Using device: {device}")

    # --- Hyperparameters ---
    batch_size = 64
    epochs = 20
    learning_rate = 0.001
    scheduler_stepsize = 10
    scheduler_gamma = 0.5

    # --- model-specific hyperparameters ---
    kernel_sizes=5
    model_config = {
    "input_sizes": (1, 64, 64),
    "convol_channels": [16, 32],
    "output_dim": 15,
    "kernel_sizes": kernel_sizes,
    "stride": 1,
    "padding": int(kernel_sizes-1)//2,
    "use_maxpooling_every": 1,
    "pooling_size": 2,
    "fc_layer_sizes": [128],
    "activation": "relu",
    "use_batchnorm": False,
    "dropout": 0.2
    }

    # -----------------------------
    # 3. Dataloaders
    # -----------------------------

    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True, seed=SEED)
    val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False, seed=SEED)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False, seed=SEED)


    # -----------------------------
    # 4. Model, Loss, Optimizer, Scheduler
    # -----------------------------

    model = CNN(**model_config).to(device)
    print(model)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}, Trainable: {num_trainable:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=scheduler_stepsize, gamma=scheduler_gamma)


    # -----------------------------
    # 5. Training Loop
    # -----------------------------

    os.makedirs("./results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    history = {
    "train_loss": [], "train_acc": [],
    "val_loss": [], "val_acc": []
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


        # Save checkpoint every few epochs (optional)
        # if epoch % 5 == 0 or epoch == epochs:
        #     checkpoint_path = f"./results/checkpoint_{timestamp}_epoch{epoch}.pt"
        #     save_checkpoint(checkpoint_path, model, optimizer, epoch=epoch, seed=SEED)


    # -----------------------------
    # 6. Save Training History & Config
    # -----------------------------

    history_df = pd.DataFrame({
        "epoch": range(1, epochs + 1),
        "train_loss": history["train_loss"],
        "train_acc": history["train_acc"],
        "val_loss": history["val_loss"],
        "val_acc": history["val_acc"]
        })
    history_filename = f"./results/training_history_{timestamp}.csv"
    history_df.to_csv(history_filename, index=False)
    print(f"\nTraining history saved to: {history_filename}")

    config_path = f"./results/model_config_{timestamp}.json"
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=4)
    print(f"Model configuration saved to: {config_path}")


    # -----------------------------
    # 7. Save final weights
    # -----------------------------

    final_weights_path = f"./results/weights_{timestamp}.pt"
    torch.save(model.state_dict(), final_weights_path)
    print(f"Model weights saved to: {final_weights_path}")

