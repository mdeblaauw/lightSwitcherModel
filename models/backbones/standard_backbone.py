import torch
from torch import nn

class Flatten(nn.Module):
    """Converts N-dimensional Tensor of shape [batch_size, d1, d2, ..., dn] to 2-dimensional Tensor
    of shape [batch_size, d1*d2*...*dn].

    # Arguments
        input: Input tensor
    """
    def forward(self, input):
        return input.view(input.size(0), -1)

def conv_1d_block(in_channels: int, out_channels:int, kernel = 3, pad = 0, conv_stride=3) -> nn.Module:

    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel, padding=pad, stride=conv_stride),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=3, stride=3)
    )

def conv_2d_block(in_channels: int, out_channels: int, kernel=32, pad=1, conv_stride=3) -> nn.Module:
    """Returns a Module that performs 3x3 convolution, ReLu activation, 2x2 max pooling.
    # Arguments
        in_channels:
        out_channels:
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel=kernel, padding=pad, stride=conv_stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=3)
    )

def get_backbone(input="1d", kernel=32, pad=0) -> nn.Module:
    if input == "1d":
        return nn.Sequential(
            conv_1d_block(1, 64, kernel, conv_stride=2),
            conv_1d_block(64, 64, kernel, conv_stride=2),
            conv_1d_block(64, 32, kernel, conv_stride=2),
            conv_1d_block(32, 32, kernel, conv_stride=1),
            Flatten()
        )
    else:
        return nn.Sequential(
            conv_2d_block(1, 64, kernel, conv_stride=2),
            conv_2d_block(64, 64, kernel, conv_stride=2),
            conv_2d_block(64, 32, kernel, conv_stride=2),
            conv_2d_block(32, 32, kernel, conv_stride=1),
            Flatten()
        )

