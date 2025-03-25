from dataclasses import dataclass
from math import log2
from typing import List

from torch import nn

from src.utils.common_utils import AppLog


@dataclass
class NewSize:
    starting_size: int
    ratio: float


@dataclass
class SquareKernel:
    kernel_size: int


@dataclass
class SeparatedKernel:
    kernel_size: int
    parameter_ratio: float
    intermediate_ratio: float


CNNKernel = SquareKernel | SeparatedKernel


# The function will make a 1×k kernel followed by a k×1 kernel with c_in^(1-a)×c_out^a intermediate channels.
# By default, c_intermediate is geometric mean of c_in and c_out i.e. a = 1/2.
# The other parameter that can be used is 'r' which is the ratio of parameters of an original k×k kernel.
# Since a k×k kernel has (k×k×c_in + 1)×c_out parameters, the number total number of parameters
# of the new two kernels will be roughly r×(k×k×c_in + 1)×c_out parameters.
# The 'a' will be calculated automatically. It will warn if the calculated 'a' falls outside [0,1].


def generate_separated_kernels(k_size: int, input_channel: int, output_channel: int, a=(1 / 2), r=0.0,
                               add_padding=True):
    c_in = input_channel
    c_out = output_channel
    t = c_out / c_in
    k = k_size

    if 0 < r < 1 and t != 1:
        a = log2(c_in * k ** 2 * r + r - 1) / log2(t) - log2(c_in * k * (t + 1) + 1) / log2(t) + 1

    c_intermediate = int(round((c_in ** (1 - a) * c_out ** a), 0))
    AppLog.info(f'c_in={c_in}, c_intermediate={c_intermediate}, c_out={c_out}')
    if not (0 <= a <= 1):
        AppLog.warn(
            f'Inconsistency in intermediate features: {c_intermediate} ∉ [{c_in},{c_out}]')

    if add_padding:
        padding = k // 2
        conv_layer_1 = nn.Conv2d(c_in, c_intermediate, (k, 1), padding=(padding, 0), bias=False)
        conv_layer_2 = nn.Conv2d(c_intermediate, c_out, (1, k), padding=(0, padding), bias=False)
        return conv_layer_1, conv_layer_2
    else:
        conv_layer_1 = nn.Conv2d(c_in, c_intermediate, (k, 1))
        conv_layer_2 = nn.Conv2d(c_intermediate, c_out, (1, k))
        return conv_layer_1, conv_layer_2


def conv_kernel(k_size: int, input_channel: int, output_channel: int, separable, ratio):
    if separable:
        if ratio > 1:
            return generate_separated_kernels(k_size, input_channel, output_channel, add_padding=False)
        else:
            return generate_separated_kernels(k_size, input_channel, output_channel, add_padding=True)
    else:
        if ratio > 1:
            return nn.Conv2d(input_channel, output_channel, kernel_size=k_size)
        else:
            return nn.Conv2d(input_channel, output_channel, kernel_size=k_size, padding=k_size // 2)


def channel_kernel_compute(inp_out_channels: List[int], layers):
    in_channel = inp_out_channels[0]
    out_channel = inp_out_channels[1]
    ratio = (out_channel / in_channel) ** (1 / layers)
    channels = [round(in_channel * ratio ** x) for x in range(layers + 1)]
    return channels


class Decoder(nn.Module):
    def __init__(self, input_size, output_size, kernel_sizes=None, channels=None, last_activation=nn.Tanh()):
        super().__init__()
        size = input_size
        layers = len(kernel_sizes)
        upscale_ratio = (output_size / size) ** (1 / layers)
        upsized_channels = [int(round(size * upscale_ratio ** (layer + 1), 0)) for layer in range(layers)]
        AppLog.info(
            f'Layers = {layers}, upsizing without kernel included = {upsized_channels}, channels = {channels}')

        sequence = nn.Sequential()
        for layer in range(layers):
            up_size = upsized_channels[layer]
            ch_in = channels[layer]
            ch_out = channels[layer + 1]
            kernel_size = kernel_sizes[layer]
            k_1 = kernel_size - 1
            upsample_layer = nn.UpsamplingBilinear2d(size=up_size + k_1)
            sequence.append(upsample_layer)
            conv_layer = nn.Conv2d(ch_in, ch_out, kernel_size,
                                   bias=False) if kernel_size <= 3 else generate_separated_kernels(
                kernel_size, ch_in, ch_out, r=9 / 25, add_padding=False)
            sequence.append(conv_layer)
            sequence.append(nn.BatchNorm2d(ch_out))
            if layer < layers - 1:
                activation_layer = nn.Mish()
                sequence.append(activation_layer)
            else:
                activation_layer = last_activation
                sequence.append(activation_layer)
        self.sequence = nn.Sequential(*sequence)

    @classmethod
    def single_kernel_decode(cls, input_size, output_size, kernel_sizes, inp_out_channels):
        layers = len(kernel_sizes)

        if len(inp_out_channels) == 2 and layers > 1:
            channels = channel_kernel_compute(inp_out_channels, layers)
        elif len(inp_out_channels) == 3 and layers > 2:
            channels_before = channel_kernel_compute(inp_out_channels[:-1], layers - 1)
            channels = [*channels_before, inp_out_channels[-1]]
        else:
            channels = [*inp_out_channels]
        print(f'Decoder channels and kernels: {channels},{kernel_sizes}')
        dec = Decoder(input_size, output_size, kernel_sizes, channels)
        return dec

    def forward(self, latent_z):
        return self.sequence.forward(latent_z)


class Encoder(nn.Module):
    def __init__(self, input_size, output_size, kernel_sizes=None, channels=None):
        super().__init__()

        layers = len(kernel_sizes)
        downscale_ratio = (output_size / input_size) ** (1 / layers)
        sequence = []
        downsampled_sizes = [int(round(input_size * downscale_ratio ** (layer + 1), 0)) for layer in range(layers)]
        AppLog.info(
            f'Layers = {layers}, downsampled_sizes = {downsampled_sizes}, channels = {channels}')

        for layer in range(layers):
            ch_in, ch_out = channels[layer], channels[layer + 1]
            kernel_size = kernel_sizes[layer]
            # padding = kernel_size // 2
            conv_layer1, conv_layer2 = generate_separated_kernels(kernel_size, ch_in, ch_out)

            # conv_layer = nn.Conv2d(ch_in, ch_out, kernel_size, padding=padding)
            activation_layer = nn.Mish()
            pooling_layer = nn.FractionalMaxPool2d(2, output_size=downsampled_sizes[layer])
            sequence.append(conv_layer1)
            sequence.append(conv_layer2)
            sequence.append(nn.BatchNorm2d(ch_out))
            sequence.append(activation_layer)
            sequence.append(pooling_layer)
        self.sequence = nn.Sequential(*sequence)

    @classmethod
    def single_kernel_encode(cls, input_size, output_size, kernel_sizes, inp_out_channels):
        layers = len(kernel_sizes)

        if len(inp_out_channels) == 2 and layers > 1:
            channels = channel_kernel_compute(inp_out_channels, layers)
        elif len(inp_out_channels) == 3 and layers > 2:
            channels_later = channel_kernel_compute(inp_out_channels[1:], layers - 1)
            channels = [inp_out_channels[0], *channels_later]
        else:
            channels = [*inp_out_channels]
        print(f'Encoder channels and kernels: {channels},{kernel_sizes}')
        enc = Encoder(input_size, output_size, kernel_sizes, channels)
        return enc

    def forward(self, input_x):
        return self.sequence.forward(input_x)
