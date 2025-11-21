import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# Same activation dictionary you used
ACTIVATIONS = {
    "sigmoid": nn.Sigmoid(),
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "leakyrelu": nn.LeakyReLU(),
    "gelu": nn.GELU(),
}

class CNN(nn.Module):
    def __init__(
        self,
        input_channels,
        conv_channels,            # e.g. [32, 64, 128]
        kernel_sizes,             # e.g. [3, 3, 3] or single int
        stride=1,
        use_padding=True,
        padding_size=1,
        use_maxpooling=True,
        use_maxpooling_every=1,
        maxpooling_size=2,
        fc_layer_sizes=[],        # e.g. [256, 128, 15]
        input_size=(64, 64),      # H, W
        activation="relu"
    ):
        super().__init__()

        # ----------------------------------------------------
        # Validate activation
        # ----------------------------------------------------
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. "
                             f"Choose from: {list(ACTIVATIONS.keys())}")
        self.activation = ACTIVATIONS[activation]

        # ----------------------------------------------------
        # Normalize kernel size list
        # ----------------------------------------------------
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(conv_channels)

        assert len(kernel_sizes) == len(conv_channels), \
            "kernel_sizes and conv_channels must have same length"

        self.convs = nn.ModuleList()
        current_c = input_channels
        h, w = input_size

        # ----------------------------------------------------
        # Build convolution + optional pooling layers
        # ----------------------------------------------------
        for i, out_c in enumerate(conv_channels):

            padding = padding_size if use_padding else 0
            ks = kernel_sizes[i]

            # Warning check for integer output
            if (h + 2 * padding - ks) % stride != 0:
                warnings.warn(
                    f"[Warning] Conv layer {i}: output height not integer. "
                    f"(h + 2*pad - ks) % stride != 0  --> "
                    f"({h} + 2*{padding} - {ks}) % {stride} != 0"
                )
            if (w + 2 * padding - ks) % stride != 0:
                warnings.warn(
                    f"[Warning] Conv layer {i}: output width not integer. "
                    f"({w} + 2*{padding} - {ks}) % {stride} != 0"
                )

            # Convolution layer
            self.convs.append(
                nn.Conv2d(current_c, out_c, kernel_size=ks, stride=stride, padding=padding)
            )

            # Update feature map size
            h = (h + 2 * padding - ks) // stride + 1
            w = (w + 2 * padding - ks) // stride + 1

            # Insert pooling if needed
            if use_maxpooling and (i + 1) % use_maxpooling_every == 0:
                self.convs.append(nn.MaxPool2d(maxpooling_size))
                h //= maxpooling_size
                w //= maxpooling_size

            current_c = out_c

        # ----------------------------------------------------
        # Flattened feature dimension after all convs
        # ----------------------------------------------------
        self.feature_dim = current_c * h * w

        # ----------------------------------------------------
        # Build fully connected layers
        # ----------------------------------------------------
        layers = []
        prev_dim = self.feature_dim

        for size in fc_layer_sizes:
            layers.append(nn.Linear(prev_dim, size))
            layers.append(self.activation)
            prev_dim = size

        self.fc = nn.Sequential(*layers) if layers else nn.Identity()

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------
    def forward(self, x):
        for layer in self.convs:
            if isinstance(layer, nn.Conv2d):
                x = self.activation(layer(x))
            else:
                x = layer(x)

        x = x.view(x.size(0), -1)
        return self.fc(x)
