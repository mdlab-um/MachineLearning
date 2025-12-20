import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import optuna
import random
import pandas as pd
import numpy as np
from datetime import datetime
import sys

# Suppress Tokenizer Parallelism Warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Project Specific Imports ---
from IMDB_ReviewDatasets import train_dataset, val_dataset, test_dataset
from IMDBtransformer import IMDBTransformerClassifier
from pytorch_utils import set_seed, get_device, get_dataloader

# --- Fixed Settings ---
BATCH_SIZE = 16 # Keep small for Transformers
EPOCHS = 5      # Reduced for search efficiency (increase for final training)
VOCAB_SIZE = 30522
MAX_LEN = 512
DEVICE = get_device()

def train_one_epoch(model, optimizer, criterion, loader):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        # 1. Unpack
        input_ids = batch['input_ids'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        hf_mask = batch['attention_mask'].to(DEVICE)

        # 2. Create Mask (True = Pad)
        padding_mask = (hf_mask == 0)

        optimizer.zero_grad()

        # 3. Forward
        outputs = model(input_ids, padding_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Stats
        running_loss += loss.item() * input_ids.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    return running_loss / total, correct / total

def evaluate(model, criterion, loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            hf_mask = batch['attention_mask'].to(DEVICE)
            padding_mask = (hf_mask == 0)

            outputs = model(input_ids, padding_mask)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * input_ids.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    return running_loss / total, correct / total

# ----------------------------
# Optuna Objective
# ----------------------------
def objective(trial):
    # 1. Suggest Hyperparameters
    # We choose powers of 2 to ensure divisibility
    d_model = trial.suggest_categorical("d_model", [128, 256])
    nhead = trial.suggest_categorical("nhead", [2, 4, 8])
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout = trial.suggest_categorical("dropout", [0.1, 0.3])
    lr = trial.suggest_loguniform("lr", 1e-5, 5e-4)

    # Validation: d_model must be divisible by nhead
    if d_model % nhead != 0:
        raise optuna.exceptions.TrialPruned(f"Invalid combo: d_model {d_model} not divisible by nhead {nhead}")

    # 2. Set Seed for Reproducibility of this trial
    # (Optional: Vary seed if you want to test stability, but fixed is usually better for HParam comparison)
    seed = 42
    set_seed(seed)

    # 3. Create DataLoaders
    # We use num_workers=0 to avoid the fork warning issues inside Optuna
    train_loader = get_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, seed=seed, num_workers=0)
    val_loader = get_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, seed=seed, num_workers=0)

    # 4. Initialize Model
    model = IMDBTransformerClassifier(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=2,
        max_len=MAX_LEN
    ).to(DEVICE)

    # Apply dropout suggestion (if your class doesn't have it in __init__, we can modify it manually)
    # Assuming your class uses self.dropout = nn.Dropout(0.1) by default:
    model.dropout.p = dropout

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 5. Training Loop
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        try:
            train_loss, train_acc = train_one_epoch(model, optimizer, criterion, train_loader)
            val_loss, val_acc = evaluate(model, criterion, val_loader)

            # Report intermediate objective value for pruning
            trial.report(val_acc, epoch)

            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if val_acc > best_val_acc:
                best_val_acc = val_acc

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Trial failed with OOM: d_model={d_model}, layers={num_layers}")
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()
            else:
                raise e

    return best_val_acc

if __name__=='__main__':
    # ----------------------------
    # Run Optuna Study
    # ----------------------------
    print(f"Starting Hyperparameter Search on {DEVICE}...")

    # Create study
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())

    # Run trials (Adjust n_trials based on how much time you have)
    study.optimize(objective, n_trials=10)

    # ----------------------------
    # Save Results
    # ----------------------------
    os.makedirs("./results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save CSV
    df = study.trials_dataframe()
    csv_path = f"./results/transformer_hypersearch_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    print("-" * 50)
    print(f"Search Complete. Results saved to: {csv_path}")
    print("Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    print(f"Best Validation Accuracy: {study.best_value:.4f}")
    print("-" * 50)
