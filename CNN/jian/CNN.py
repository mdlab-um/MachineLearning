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


class CNN(nn.Module):
    def __init__(self, input_sizes=(1,64,64), # one color channel
                 convol_channels=[16, 32], # list of channel numbers for multiple convolutional layer
                 output_dim=15, # output classes
                 kernel_sizes=3,
                 stride=1,
                 # padding: 0 means no padding; 1 means padding size = 1; so on so forth; 'auto' means (F-1)/S
                 padding=0,
                 use_maxpooling_every=0, # 0 means no maxpooling; 1: one conv layer + one maxpooling layer
                 pooling_size=2,
                 fc_layer_sizes=[],  # e.g. [256, 128, 15]
                 activation="relu",
                 use_batchnorm=False,
                 dropout=0,
            ):
        super().__init__()

        # activate name check
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation}")

        # normalize kernel sizes
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(convol_channels)

        if isinstance(padding,int ):
            padding = [padding] * len(convol_channels)

        prev_c, h, w = input_sizes
        layers = []
        for i,(current_c,ks,pd) in enumerate(zip(convol_channels, kernel_sizes, padding), 1):
            # convoluational layer
            layers.append(nn.Conv2d(prev_c, current_c, kernel_size=ks,\
                                    stride=stride, padding=pd))
            # batch normalization
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(current_c))
            # activation func.
            layers.append(ACTIVATIONS[activation]())

            # update feature map size
            h = (h + 2 * pd - ks) // stride + 1
            w = (w + 2 * pd - ks) // stride + 1

            if use_maxpooling_every > 0:
                if i % use_maxpooling_every == 0:
                    layers.append(nn.MaxPool2d(pooling_size))
                    h //= pooling_size
                    w //= pooling_size

            prev_c = current_c

        self.conv_blocks = nn.Sequential(*layers)

        # final feature map dimension
        self.feature_dim = prev_c * h * w

        # feed-forward fully connected network
        fc_layers = []
        prev_dim = self.feature_dim
        for s in fc_layer_sizes:
            fc_layers.append(nn.Linear(prev_dim, s))
            fc_layers.append(ACTIVATIONS[activation]())
            fc_layers.append(nn.Dropout(dropout))
            prev_dim = s

        # output layer
        fc_layers.append(nn.Linear(prev_dim, output_dim))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

