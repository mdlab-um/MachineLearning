import torch
import torch.nn as nn
import torch.nn.functional as F

# Fully connected neural network
ACTIVATIONS = {
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU,
}


class FNN(nn.Module):
    def __init__(self, input_dim=64, layer_sizes=[64, 128],\
                output_dim=15, activation="sigmoid", dropout=0.4):
        super().__init__()

        # activate name check
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []
        prev_dim = input_dim
        for s in layer_sizes:
            layers.append(nn.Linear(prev_dim, s))
            layers.append(ACTIVATIONS[activation]())
            layers.append(nn.Dropout(dropout))
            prev_dim = s

        # output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        # flaten to one-dimension
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


