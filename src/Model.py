import torch
from torch import nn
from torch.version import cuda

from src.codec import Encoder, Decoder
DEVICE = torch.device("cuda" if cuda else "cpu")

class Model(nn.Module):
    def __init__(self,latent_channels,latent_size,input_channels,output_channels,size):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.size = size
        self.encoder = Encoder((latent_channels,latent_size),(input_channels,size),5,3)
        self.decoder = Decoder((latent_channels,latent_size),(output_channels,size),5,3)
        self.variance_upsample = nn.Upsample(size=(latent_size, latent_size))


    @staticmethod
    def reparameterization(mean, var):
        epsilon = torch.randn_like(mean) # sampling epsilon

        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self,x):
        mu,log_var = self.encoder.forward(x)
        var = torch.exp(0.5 * log_var)
        var_up = self.variance_upsample(var)
        z_sample = self.reparameterization(mu, var_up)
        decoded = self.decoder(z_sample)
        return mu, var, log_var, decoded

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False

    def unfreeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = True

    def generate(self,given_sample):
        mu, log_var = self.encoder.forward(given_sample)
        var = torch.exp(0.5 * log_var)
        def get_new_sample():
            z_sample = self.reparameterization(mu, var)
            decoded = self.decoder(z_sample)
            return z_sample, decoded
        return get_new_sample()