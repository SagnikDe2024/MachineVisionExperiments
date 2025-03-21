from typing import List

from torch import nn


class Rotate(nn.Module):
    def __init__(self, downscale : List[int],kernels : List[int], channels : List[int]):
        super().__init__()
        layers = len(kernels)
        rotation = 90/layers
        downscale_raw = []
        if len(downscale) == 1:
            ratio = downscale[0]
            downscale_raw = [ratio for _ in range(layers)]
        else:
            downscale_raw = [*downscale]

        for k, kernel_size in enumerate(kernels):
            padding = kernel_size // 2
            cnn = nn.Conv2d(channels[k], channels[k+1], kernel_size, padding=padding)
            activation = nn.Mish()
            pool = nn.MaxPool2d(2, 2)



