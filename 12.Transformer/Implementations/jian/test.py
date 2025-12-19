import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import sys
import os

# --- Imports tailored to your new project structure ---
from IMDB_ReviewDatasets import test_dataset
from IMDBtransformer import IMDBTransformerClassifier
from pytorch_utils import get_dataloader

# --- 1. Load config & weights from command line ---
if len(sys.argv) < 3:
    print("Usage: python test.py <config_path> <weights_path>")
    sys.exit(1)

config_path = sys.argv[1]
weights_path = sys.argv[2]

with open(config_path, "r") as f:
    cfg = json.load(f)

# --- 2. Initialize Model ---
# We unpack the config dictionary directly into the Transformer class
model = IMDBTransformerClassifier(**cfg)

# --- 3. Load Weights ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()

# --- Print Params ---
num_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {num_params:,}")


def get_predictions_and_confusion(model, test_dataset, batch_size=16):
    """
    Get predictions for the entire test dataset and plot a binary confusion matrix.
    """
    # Use the helper to get a loader (handles collation if needed)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    all_labels = []
    all_preds = []
    test_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    print("Running evaluation...")

    with torch.no_grad():
        for batch in test_loader:
            # 1. Unpack Batch (Dictionary from HuggingFace)
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            hf_mask = batch['attention_mask'].to(device)

            # 2. Generate Padding Mask (True where padding exists)
            padding_mask = (hf_mask == 0)

            # 3. Forward Pass
            outputs = model(input_ids, padding_mask)

            # 4. Metrics
            loss = criterion(outputs, labels)
            test_loss += loss.item() * input_ids.size(0)

            _, preds = outputs.max(1)

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    # Calculate final metrics
    test_loss /= total
    test_acc = correct / total

    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Concatenate all batches
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    # --- Confusion Matrix (Binary: 0=Negative, 1=Positive) ---
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    # Plot
    plt.figure(figsize=(6, 5))

    # Define labels for the plot
    class_names = ['Negative', 'Positive']

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix\nAccuracy: {test_acc:.2%}")

    # Save output
    os.makedirs('./results', exist_ok=True)
    save_path = './results/testing_confmat.png'
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.show()

    return all_labels, all_preds, cm

# Run the evaluation
if __name__ == "__main__":
    all_labels, all_preds, cm = get_predictions_and_confusion(model, test_dataset)
