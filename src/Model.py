import logging
from math import floor, log2
from typing import List

import torch
from torch import nn
from torchinfo import summary

from src.codec import Encoder, Decoder

logger = logging.getLogger(__name__)

logging.basicConfig(filename='../log/ImageEncDec.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s',
                    encoding='utf-8', level=logging.INFO)

logger.setLevel(logging.INFO)

class Model(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def reparameterization(mean, log_var):
        var = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(var)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, x):
        mu, log_var = self.encoder.forward(x)
        z_sample = self.reparameterization(mu, log_var)
        decoded = self.decoder.forward(z_sample)

        # 2 way regularization, a latent going through decoder -> encoder chain should have the same mean as the one generated.
        z_sample_2 = self.reparameterization(mu, log_var)
        decoded_2 = self.decoder.forward(z_sample_2)
        z_sample_2_mu, _ = self.encoder.forward(decoded_2)

        return mu, log_var, decoded, z_sample_2_mu

    # I believe that the decoder can be fine-tuned with generated latents and so the encoder is frozen
    def encoder_eval_mode(self):
        self.encoder.eval()

    # Generate sample like a given sample
    def generate(self, given_sample):
        mu, log_var = self.encoder.forward(given_sample)

        def get_new_sample():
            z_sample = self.reparameterization(mu, log_var)
            decoded = self.decoder(z_sample)
            return z_sample, decoded

        return get_new_sample()


# This will classify the CIFAR-10 model into the classes for now. Might be helpful for checking the generation
# when it will be used later.

class Classifier(nn.Module):
    def __init__(self, dnn_layers: List[int], starting_size=32, feature_upscale=4 / 3):
        super().__init__()

        final_downscaled_size = 2
        final_channels = dnn_layers[0] / (final_downscaled_size ** 2)
        image_downscale = 2 / 3

        layers_required = floor(log2(final_downscaled_size / starting_size) / log2(image_downscale))
        channels_rest = [round(final_channels * (1 / feature_upscale) ** l) for l in range(layers_required)]
        channels_rest.reverse()
        channels = [3, *channels_rest]
        kernels = [3 for _ in range(layers_required)]
        encoder = Encoder(starting_size, final_downscaled_size, kernels, channels)

        self.encoder = encoder
        sequence = nn.Sequential()
        sequence.append(nn.Mish())
        # nn.AdaptiveAvgPool2d((1,1))
        # Flatten the result from the encoder first
        sequence.append(nn.Flatten())

        for layer_dnn in range(len(dnn_layers) - 2):
            # Apply an activation
            act = nn.Mish()
            lin = nn.Linear(dnn_layers[layer_dnn], dnn_layers[layer_dnn + 1])
            sequence.append(lin)
            sequence.append(act)
        sequence.append(nn.Linear(dnn_layers[-2], dnn_layers[-1]))
        # sequence.append(nn.Softmax(dim=1)) No need as the cross-entropy loss will normalize on its own.
        self.sequence = nn.Sequential(*sequence)
        self.normalized = nn.Softmax(dim=1)

    def forward(self, x):
        features = self.encoder.forward(x)
        raw_probability_values = self.sequence(features)
        probabilities = self.normalized(raw_probability_values)
        # Normalize when using the model.
        return raw_probability_values, probabilities


if __name__ == '__main__':
    classifier = Classifier([96, 50, 10], 32, 4 / 3)

    summary(classifier, input_size=(128, 3, 32, 32))
