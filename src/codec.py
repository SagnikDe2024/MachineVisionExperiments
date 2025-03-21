import logging
from typing import List

from torch import nn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# A k x k kernel with c_in input channels and c_out output channels will have params = (k*k*c_in + 1)*c_out
# This will create two kernels 1 x k and k x 1 with intermediate number of features c_intermediate so that it has  (ratio_to_k2) * params parameters


def generate_separated_kernels(k_size: int, input_channel: int, output_channel: int, ratio_to_k2: float,
                               add_padding=True):
    r = ratio_to_k2
    c_in = input_channel
    c_out = output_channel
    k = k_size
    c_intermediate = round((c_out * (-1 + r + c_in * (k ** 2) * r)) / (1 + (c_in + c_out) * k), 0)
    if c_in != c_out:
        intermediate_location = (c_intermediate - c_in) / (c_out - c_in)
        if 0 <= intermediate_location <= 1:
            logger.warn(
                f'Inconsistency in intermediate features: c_in={c_in}, c_intermediate={c_intermediate}, c_out={c_out}. {c_intermediate} ∉ [{c_in},{c_out}]')
    if add_padding:
        padding = k // 2
        conv_layer_1 = nn.Conv2d(c_in, c_intermediate, (k, 1), padding=(padding, 0))
        conv_layer_2 = nn.Conv2d(c_intermediate, c_out, (1, k), padding=(0, padding))
        return nn.Sequential(conv_layer_1, conv_layer_2)
    else:
        conv_layer_1 = nn.Conv2d(c_in, c_intermediate, (k, 1))
        conv_layer_2 = nn.Conv2d(c_intermediate, c_out, (1, k))
        return nn.Sequential(conv_layer_1, conv_layer_2)


def channel_kernel_compute(inp_out_channels: List[int], layers):
    in_channel = inp_out_channels[0]
    out_channel = inp_out_channels[1]
    ratio = (out_channel / in_channel) ** (1 / layers)
    channels = [round(in_channel * ratio ** x) for x in range(layers + 1)]
    return channels


class Decoder(nn.Module):
    def __init__(self, input_size, output_size, kernel_sizes=None, channels=None):
        super().__init__()
        size = input_size
        layers = len(kernel_sizes)
        upscale_ratio = (output_size / size) ** (1 / layers)

        sequence = nn.Sequential()
        for layer in range(layers):
            up_size = int(size * upscale_ratio ** (layer + 1))
            ch_in = channels[layer]
            ch_next = channels[layer + 1]
            kernel_size = kernel_sizes[layer]
            k_1 = kernel_size - 1
            upsample_layer = nn.UpsamplingBilinear2d(size=(up_size + k_1, up_size + k_1))
            sequence.append(upsample_layer)
            conv_layer = nn.Conv2d(ch_in, ch_next, kernel_size)
            sequence.append(conv_layer)
            if layer < layers - 1:
                activation_layer = nn.Mish()
                sequence.append(activation_layer)
                # sequence.append(nn.BatchNorm2d(ch_next))
            else:
                activation_layer = nn.Tanh()
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
        sequence = nn.Sequential()
        for layer in range(layers):
            chin, chout = channels[layer], channels[layer + 1]
            kernel_size = kernel_sizes[layer]
            # padding = kernel_size // 2
            conv_layer = generate_separated_kernels(kernel_size, chin, chout, 1.125 * 2 / kernel_size)

            # conv_layer = nn.Conv2d(chin, chout, kernel_size, padding=padding)
            activation_layer = nn.Mish()
            pooling_layer = nn.FractionalMaxPool2d(2, output_ratio=downscale_ratio)
            sequence.append(conv_layer)
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
        layered_result = self.sequence.forward(input_x)
        (n, c, h, w) = layered_result.size()
        mean = layered_result[:, :c // 2, :, :]
        log_var = layered_result[:, c // 2:, :, :]
        return mean, log_var
