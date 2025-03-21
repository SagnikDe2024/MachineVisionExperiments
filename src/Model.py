from math import floor, log2
from typing import List

import torch
from torch import nn

from src.codec import Encoder, Decoder


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


class Classifier(nn.Module):
    def __init__(self, dnn_layers: List[int], starting_size=32, feature_upscale=4 / 3):
        super().__init__()

        final_downscaled_size = 1
        final_channels = dnn_layers[0]
        image_downscale = 3 / 5

        layers_required = floor(log2(final_downscaled_size / starting_size) / log2(image_downscale))
        channels_rest = [round(final_channels * (1 / feature_upscale) ** l) for l in range(layers_required)]
        channels_rest.reverse()
        channels = [3, *channels_rest]
        kernels = [3 for _ in range(layers_required)]
        encoder = Encoder(starting_size, 1, kernels, channels)

        self.encoder = encoder
        sequence = nn.Sequential()

        # Flatten the result from the encoder first
        sequence.append(nn.Flatten())

        for layer_dnn in range(len(dnn_layers) - 2):
            # Apply an activation
            act = nn.Mish()
            lin = nn.Linear(dnn_layers[layer_dnn], dnn_layers[layer_dnn + 1])
            sequence.append(lin)
            sequence.append(act)
        sequence.append(nn.Linear(dnn_layers[-2], dnn_layers[-1]))
        sequence.append(nn.Softmax(dim=1))
        self.sequence = nn.Sequential(*sequence)

    def forward(self, x):
        features = self.encoder.forward(x)
        probabilities = self.sequence(features)
        return probabilities


def get_children(model: torch.nn.Module):
    # get children form model
    children = list(model.children())
    flat_children = []
    if not children:
        # if model has no children; model is last child
        return model
    else:
        # look for children from children, to the last child
        for child in children:
            try:
                flat_children.extend(get_children(child))
            except TypeError:
                flat_children.append(get_children(child))
    return flat_children


if __name__ == '__main__':
    classifier = Classifier([384, 62, 10], 32, 3 / 2)
    print(classifier)
    all_children = get_children(classifier)
    print(all_children)
