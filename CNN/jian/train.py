import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import warnings
import random
from CNN_module import CNN, ACTIVATIONS
from datasets import train_dataset, val_dataset, test_dataset, device


###### training
seed = 42
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Hyperparameters ---
batch_size = 32
epochs = 30
learning_rate = 0.001
scheduler_stepsize = 10
scheduler_gamma = 0.5

# --- DataLoaders ---
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# --- model-specific hyperparameters ---
input_channels=1
conv_channels=[8, 16, 32]
kernel_sizes=5
use_padding=True
padding_size=int((kernel_sizes-1) // 2)
use_maxpooling=True
use_maxpooling_every=1
fc_layer_sizes=[64, 15] # output 15 class
input_size=(64, 64)
activation="relu"
use_batchnorm=False
dropout=0.4

# --- Model ---
model = CNN(
    input_channels=input_channels,
    conv_channels=conv_channels,
    kernel_sizes=kernel_sizes,
    use_padding=use_padding,
    padding_size=padding_size,
    use_maxpooling=use_maxpooling,
    use_maxpooling_every=use_maxpooling_every,
    fc_layer_sizes=fc_layer_sizes,
    input_size=input_size,
    activation=activation,
    dropout=dropout
)
model.to(device)
print(model)

# --- Loss and optimizer ---
criterion = nn.CrossEntropyLoss()  # good for multi-class classification
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=scheduler_stepsize, gamma=scheduler_gamma)


# --- Training loop ---
train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    scheduler.step()
    train_loss = running_loss / total
    train_acc = correct / total
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)


    # --- Validation / Testing ---
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= total
    val_acc = correct / total
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)

    print(f"Epoch [{epoch}/{epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# save results:
# Save training history to CSV
history_df = pd.DataFrame({
    'epoch': range(1, epochs + 1),
    'train_loss': train_loss_history,
    'train_acc': train_acc_history,
    'val_loss': val_loss_history,
    'val_acc': val_acc_history
})

# Create a timestamp for unique filenames
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
history_filename = f"./results/training_history_{timestamp}.csv"
history_df.to_csv(history_filename, index=False)
print(f"\nTraining history saved to: {history_filename}")


# save weights
os.makedirs("./results", exist_ok=True)
save_path = f"./results/weights_{timestamp}.pt"
torch.save(model.state_dict(), save_path)
print(f"Model weights saved to: {save_path}")


# save model config
import json
config = {
    "input_channels": input_channels,
    "conv_channels": conv_channels,
    "kernel_sizes": kernel_sizes,
    "use_padding": use_padding,
    "padding_size": padding_size,
    "use_maxpooling": use_maxpooling,
    "use_maxpooling_every": use_maxpooling_every,
    "maxpooling_size": 2,
    "fc_layer_sizes": fc_layer_sizes,
    "input_size": list(input_size),
    "activation": activation,
    'use_batchnorm': use_batchnorm,
    "dropout": dropout
}

config_path = f"./results/model_config_{timestamp}.json"
with open(config_path, "w") as f:
    json.dump(config, f, indent=4)
print(f"Model configuration saved to: {config_path}")
