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


    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)  # sampling epsilon
        var_up = self.variance_upsample(var)
        z = mean + var_up * epsilon  # reparameterization trick
        return z

    def forward(self,x):
        mu,log_var = self.encoder.forward(x)
        var = torch.exp(log_var)
        z_sample = self.reparameterization(mu, var)
        decoded = self.decoder(z_sample)
        return mu, var, z_sample, decoded