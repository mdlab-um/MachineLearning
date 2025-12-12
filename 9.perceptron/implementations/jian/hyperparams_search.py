import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
import optuna
import random
import os
import pandas as pd
import numpy as np
from FNN import FNN, ACTIVATIONS
from datasets import train_dataset, val_dataset, test_dataset, device
from datetime import datetime

###### training
# --- fixed Hyperparameters ---
batch_size = 64
epochs = 10
learning_rate = 0.001
scheduler_stepsize = 10
scheduler_gamma = 0.5

def train_one_epoch(model, optimizer, criterion, loader):
    model.train()
    running_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, criterion, loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            running_loss += loss.item() * X.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
    return running_loss / len(loader.dataset), correct / len(loader.dataset)

# ----------------------------
# Optuna objective
# ----------------------------
def objective(trial):
    # Hyperparameter search space
    seed = trial.suggest_int("seed", 1, 10000)
    layer1 = trial.suggest_categorical("layer1", [64, 128, 256])
    layer2 = trial.suggest_categorical("layer2", [64, 128, 256])
    dropout = trial.suggest_categorical("dropout", [0.2, 0.4, 0.6])
    # lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    activation = trial.suggest_categorical("activation", ["relu", "leakyrelu", "sigmoid"])

    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # RECREATE DATALOADERS with seeded generator
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=0, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    model = FNN(input_dim=64*64,  # assuming 64x64 images
                layer_sizes=[layer1, layer2],
                output_dim=15,
                activation=activation,
                dropout=dropout).to(device)

    # --- Loss and optimizer ---
    criterion = nn.CrossEntropyLoss()  # good for multi-class classification
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=scheduler_stepsize, gamma=scheduler_gamma)

    # Train for few epochs
    # epochs = 15  # keep small for quick search
    for epoch in range(epochs):
        train_one_epoch(model, optimizer, criterion, train_loader)
        scheduler.step()

    val_loss, val_acc = evaluate(model, criterion, val_loader)
    return val_acc  # Optuna minimizes objective


if __name__=='__main__':
    # ----------------------------
    # Run Optuna study
    # ----------------------------
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)  # adjust n_trials

    # ----------------------------
    # Save results to CSV
    # ----------------------------
    os.makedirs("./results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = study.trials_dataframe()
    df.to_csv(f"./results/hypersearch_results_{timestamp}.csv", index=False)
    print(f"Saved: ./results/hypersearch_results_{timestamp}.csv")
    print("Best hyperparameters:", study.best_trial.params)
    print("Best validation loss:", study.best_trial.value)

