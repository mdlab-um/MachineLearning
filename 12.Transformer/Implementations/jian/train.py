import os
# tell the tokenizers library (from huggingface) to stop using threads
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm # Added for progress visibility

import pandas as pd
from IMDB_ReviewDatasets import train_dataset, val_dataset, test_dataset
from pytorch_utils import set_seed, get_device, get_dataloader, save_checkpoint
from IMDBtransformer import IMDBTransformerClassifier

# -----------------------------
# 0. Training / Evaluation Functions
# -----------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Wrapped in tqdm for a progress bar
    loop = tqdm(loader, leave=False)

    # CHANGED: Loader now returns a dictionary (batch), not (images, labels)
    for batch in loop:
        # 1. Unpack the batch
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        hf_mask = batch['attention_mask'].to(device) # 1=Real, 0=Pad

        # 2. Create the Padding Mask for PyTorch Transformer
        # PyTorch expects: True = Ignore (Pad), False = Attend (Real Word)
        padding_mask = (hf_mask == 0)

        optimizer.zero_grad()

        # 3. Pass both Input IDs and the converted Mask
        outputs = model(input_ids, padding_mask)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update stats
        running_loss += loss.item() * input_ids.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        loop.set_description(f"Loss: {loss.item():.4f}")

    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            # 1. Unpack
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            hf_mask = batch['attention_mask'].to(device)

            # 2. Create Mask
            padding_mask = (hf_mask == 0)

            # 3. Forward
            outputs = model(input_ids, padding_mask)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * input_ids.size(0)
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
    epochs = 10
    learning_rate = 1e-3 # Reduced LR is usually better for Transformers
    scheduler_stepsize = 10
    scheduler_gamma = 0.5

    # --- model-specific hyperparameters ---
    # kernel_sizes=5 # Removed (CNN specific)
    model_config = {
        "vocab_size": 30522, # Standard BERT vocab size
        "d_model": 256,      # Embedding dimension
        "nhead": 4,          # Attention heads
        "num_layers": 2,     # Encoder layers
        "num_classes": 2,    # Positive/Negative
        "max_len": 512,      # Max sequence length
    }

    # -----------------------------
    # 3. Dataloaders
    # -----------------------------

    # Ensure get_dataloader supports Hugging Face datasets (which yield dicts)
    # If using the standard DataLoader(dataset, ...), this works automatically.
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True, seed=SEED)
    val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False, seed=SEED)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False, seed=SEED)


    # -----------------------------
    # 4. Model, Loss, Optimizer, Scheduler
    # -----------------------------

    model = IMDBTransformerClassifier(**model_config).to(device)
    # print(model) # Optional: reduced clutter

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

    print("Starting training...")
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


        # Save checkpoint (Optional: uncomment if needed)
        # if epoch % 5 == 0:
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
