import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datasets import test_dataset, device
from FNN import FNN, ACTIVATIONS
import json
import sys

# --- Load config ---
with open(sys.argv[1], "r") as f:
    cfg = json.load(f)

model = FNN(
    input_dim=cfg["input_dim"],
    layer_sizes=cfg["layer_sizes"],
    output_dim=cfg["output_dim"],
    activation=cfg["activation"],
    dropout=cfg["dropout"]
)

# -- Load weights --
model.load_state_dict(torch.load(sys.argv[2], map_location="cpu"))

# --- Number of parameters ---
num_params = sum(p.numel() for p in model.parameters())
num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {num_params:,}")
print(f"Trainable parameters: {num_trainable:,}")


def get_predictions_and_confusion(model, test_dataset, batch_size=64):
    """
    Get predictions for the entire test dataset and plot a confusion matrix.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_labels = []
    all_preds = []
    test_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    test_loss /= total
    test_acc = correct / total

    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Concatenate all batches
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    # print(all_labels.shape)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(15))

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.arange(15), yticklabels=np.arange(15))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    return all_labels, all_preds, cm

all_labels, all_preds, cm = get_predictions_and_confusion(model, test_dataset)
