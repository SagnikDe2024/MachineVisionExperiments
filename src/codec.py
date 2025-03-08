from torch import nn


class Decoder(nn.Module):
    def __init__(self, latent_size, output_size, kernel_size=5, layers=3):
        super().__init__()
        (ch, size) = latent_size
        (ch_o, size_out) = output_size
        upscale_ratio = (size_out / size) ** (1 / layers)
        k_1 = kernel_size - 1
        sequence = nn.Sequential()
        for layer in range(layers):
            size = size * upscale_ratio ** (layer + 1)
            ch_in = ch / (upscale_ratio ** layer)
            ch_next = ch / (upscale_ratio ** (layer + 1))
            upsample_layer = nn.UpsamplingBilinear2d(size=(size + k_1, size + k_1))
            sequence.append(upsample_layer)
            if layer < layers - 1:
                conv_layer = nn.Conv2d(ch_in, ch_next, kernel_size)
                activation_layer = nn.Mish()
                sequence.append(conv_layer)
                sequence.append(activation_layer)
                sequence.append(nn.BatchNorm2d(ch_next))
            else:
                conv_layer = nn.Conv2d(ch_in, 3, kernel_size)
                activation_layer = nn.Sigmoid()
                sequence.append(conv_layer)
                sequence.append(activation_layer)
        self.sequence = nn.Sequential(*sequence)

    def forward(self, latent_z):
        return self.decoder(latent_z)


class Encoder(nn.Module):
    def __init__(self, latent_size, input_size, kernel_size=5, layers=3):
        super().__init__()
        (ch, size) = latent_size
        (ch_inp, size_in) = input_size
        downscale_ratio = (size / size_in) ** (1 / layers)
        padding = kernel_size // 2
        sequence = nn.Sequential()
        ch_in = 3
        for layer in range(layers):
            ch_next = ch / (downscale_ratio ** (layers - layer))
            conv_layer = nn.Conv2d(ch_in, ch_next, kernel_size, padding=padding)
            activation_layer = nn.Mish()
            pooling_layer = nn.FractionalMaxPool2d(kernel_size, downscale_ratio)
            ch_in = ch_next
            sequence.append(conv_layer)
            sequence.append(activation_layer)
            sequence.append(pooling_layer)
        self.sequence = nn.Sequential(*sequence)
        self.mean_activation = nn.Tanh()
        self.std_normalization = nn.BatchNorm2d(ch)
        self.std_pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, input_x):
        layered_result = self.sequence(input_x)
        mean = self.mean_activation(layered_result)
        std_1 = self.std_normalization(layered_result)
        std = self.std_pooling(std_1)
        return mean, std
