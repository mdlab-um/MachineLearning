import torch
import torch.nn as nn
import warnings

ACTIVATIONS = {
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU,
}


# ------------------------------------------------------------
# A clean block: Conv → (BatchNorm) → Activation
# ------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, ks, stride, padding, activation, use_bn=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=ks, stride=stride, padding=padding)
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(ACTIVATIONS[activation]())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ------------------------------------------------------------
# Main CNN Class
# ------------------------------------------------------------
class CNN(nn.Module):
    def __init__(
        self,
        input_channels,
        conv_channels,              # e.g., [32, 64, 128]
        kernel_sizes,               # e.g., [3,3,3] or int
        stride=1,
        use_padding=True,
        padding_size=1,
        use_maxpooling=True,
        use_maxpooling_every=1,
        maxpooling_size=2,
        fc_layer_sizes=[],          # e.g. [256, 128, 15]
        input_size=(64, 64),
        activation="relu",
        use_batchnorm=False,
        dropout=0.2
    ):
        super().__init__()

        # activate name check
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation}")

        # normalize kernel sizes
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(conv_channels)

        assert len(kernel_sizes) == len(conv_channels)

        h, w = input_size
        in_c = input_channels

        conv_blocks = []

        # ----------------------------------------------------
        # Build convolutional stack
        # ----------------------------------------------------
        for i, out_c in enumerate(conv_channels):
            ks = kernel_sizes[i]
            pad = padding_size if use_padding else 0

            # sanity check
            if (h + 2 * pad - ks) % stride != 0:
                warnings.warn(
                    f"Conv layer {i}: output height not integer. ({h}+2*{pad}-{ks}) mod {stride} != 0"
                )
            if (w + 2 * pad - ks) % stride != 0:
                warnings.warn(
                    f"Conv layer {i}: output width not integer. ({w}+2*{pad}-{ks}) mod {stride} != 0"
                )

            # Conv → BN → Activation
            conv_blocks.append(
                ConvBlock(in_c, out_c, ks, stride, pad, activation, use_bn=use_batchnorm)
            )

            # update feature map size
            h = (h + 2 * pad - ks) // stride + 1
            w = (w + 2 * pad - ks) // stride + 1

            # maxpool if needed
            if use_maxpooling and (i + 1) % use_maxpooling_every == 0:
                conv_blocks.append(nn.MaxPool2d(maxpooling_size))
                h //= maxpooling_size
                w //= maxpooling_size

            in_c = out_c

        self.conv_layers = nn.Sequential(*conv_blocks)

        # ----------------------------------------------------
        # Compute flattened feature dimension
        # ----------------------------------------------------
        self.feature_dim = in_c * h * w

        # ----------------------------------------------------
        # Build fully connected network
        # ----------------------------------------------------
        fc_layers = []
        prev = self.feature_dim

        for size in fc_layer_sizes[:-1]:      # hidden layers
            fc_layers.append(nn.Linear(prev, size))
            if dropout > 0:
                fc_layers.append(nn.Dropout(dropout))
            fc_layers.append(ACTIVATIONS[activation]())
            prev = size

        # final classifier (no activation here!)
        if len(fc_layer_sizes) > 0:
            fc_layers.append(nn.Linear(prev, fc_layer_sizes[-1]))

        self.fc = nn.Sequential(*fc_layers) if fc_layers else nn.Identity()

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
