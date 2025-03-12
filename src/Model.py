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
