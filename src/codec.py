from typing import List

from torch import nn


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
            if layer < layers - 1:
                conv_layer = nn.Conv2d(ch_in, ch_next, kernel_size)
                activation_layer = nn.Mish()
                sequence.append(conv_layer)
                sequence.append(activation_layer)
                # sequence.append(nn.BatchNorm2d(ch_next))
            else:
                conv_layer = nn.Conv2d(ch_in, 3, kernel_size)
                activation_layer = nn.Tanh()
                sequence.append(conv_layer)
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
        return self.decoder(latent_z)


class Encoder(nn.Module):
    def __init__(self, input_size, output_size, kernel_sizes=None, channels=None):
        super().__init__()

        layers = len(kernel_sizes)
        downscale_ratio = (input_size / output_size) ** (1 / layers)
        sequence = nn.Sequential()
        for layer in range(layers):
            chin, chout = channels[layer], channels[layer + 1]
            kernel_size = kernel_sizes[layer]
            padding = kernel_size // 2
            conv_layer = nn.Conv2d(chin, chout, kernel_size, padding=padding)
            activation_layer = nn.Mish()
            pooling_layer = nn.FractionalMaxPool2d(kernel_size, downscale_ratio)
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
        layered_result = self.sequence(input_x)
        (N, C, H, W) = layered_result.size()
        mean = layered_result.view[:, :C // 2, :, :]
        log_var = layered_result.view[:, C // 2:, :, :]
        return mean, log_var
