import torch
from numpy import log2
from torch import nn

from ml_common.common_utils import AppLog


def generate_separated_kernels(input_channel: int, output_channel: int, k_size: int, a: float = (1 / 2), r: float = 0.0,
							   add_padding: bool = True, bias=False, stride: int = 1):
	c_in = input_channel
	c_out = output_channel
	t = c_out / c_in
	k = k_size

	if 0 < r < 1 and t != 1:
		a = log2((t + 1) / (t * r * k)) / log2(1 / t)

	c_intermediate = int(round((c_in ** (1 - a) * c_out ** a), 0))
	AppLog.info(f'c_in={c_in}, c_intermediate={c_intermediate}, c_out={c_out}')
	if not (0 <= a <= 1):
		AppLog.warning(
				f'Inconsistency in intermediate features: {c_intermediate} ∉ [{c_in},{c_out}]')
	padding = k // 2 if add_padding else 0

	conv_layer_1 = nn.Conv2d(c_in, c_intermediate, (1, k), padding=(0, padding), bias=bias, stride=(1, stride))
	conv_layer_2 = nn.Conv2d(c_intermediate, c_out, (k, 1), padding=(padding, 0), bias=bias, stride=(stride, 1))

	return conv_layer_1, conv_layer_2


class InputLayer(nn.Module):
	def __init__(self, in_channel, out_channel):
		super().__init__()
		self.layer_1_1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2,
								   padding=1,
								   bias=False)
		layer_1_2_conv1, layer_1_2_conv2 = generate_separated_kernels(in_channel, out_channel, k_size=5, stride=2,
																	  r=9 / 25)
		self.layer_1_2_conv1 = layer_1_2_conv1
		self.layer_1_2_conv2 = layer_1_2_conv2
		self.norm = nn.BatchNorm2d(out_channel * 2)
		self.activation = nn.Mish()

	def forward(self, x):
		x3 = self.layer_1_1.forward(x)
		x5_1 = self.layer_1_2_conv1.forward(x)
		x5_2 = self.layer_1_2_conv2.forward(x5_1)
		x3and5 = torch.concat([x3, x5_2], 1)
		normed = self.norm.forward(x3and5)
		# reduced = self.reduce_conv.forward(normed)
		activated = self.activation.forward(normed)
		return activated


class MiddleLayer(nn.Module):
	def __init__(self, in_channel, out_channel):
		super().__init__()
		layer_1_1_conv1, layer_1_1_conv2 = generate_separated_kernels(in_channel, out_channel, k_size=3, stride=2,
																	  r=2 / 3)
		self.layer_1_1_conv1 = layer_1_1_conv1
		self.layer_1_1_conv2 = layer_1_1_conv2
		# self.layer_1_1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2,
		# padding=1,bias=False)
		layer_1_2_conv1, layer_1_2_conv2 = generate_separated_kernels(in_channel, out_channel, k_size=5, stride=2,
																	  r=9 / 25)
		self.layer_1_2_conv1 = layer_1_2_conv1
		self.layer_1_2_conv2 = layer_1_2_conv2
		self.norm = nn.BatchNorm2d(out_channel * 2)
		self.activation = nn.Mish()

	def forward(self, x):
		x3_1 = self.layer_1_1_conv1.forward(x)
		x3_2 = self.layer_1_1_conv2.forward(x3_1)
		x5_1 = self.layer_1_2_conv1.forward(x)
		x5_2 = self.layer_1_2_conv2.forward(x5_1)
		x3and5 = torch.concat([x3_2, x5_2], 1)
		normed = self.norm.forward(x3and5)
		activated = self.activation.forward(normed)
		return activated


class EncoderAux(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.input_layer = InputLayer(in_channel=channels[0], out_channel=channels[1])
		self.reduce_conv_input = nn.Conv2d(in_channels=channels[1] * 2, out_channels=(channels[1]), kernel_size=1,
										   stride=1,
										   padding=0)
		self.middle_layer_1 = MiddleLayer(in_channel=channels[1], out_channel=channels[2])
		self.reduce_conv_1 = nn.Conv2d(in_channels=channels[2] * 2, out_channels=(channels[2]), kernel_size=1,
									   stride=1,
									   padding=0)
		self.middle_layer_2 = MiddleLayer(in_channel=channels[2], out_channel=channels[3])
		self.reduce_conv_2 = nn.Conv2d(in_channels=channels[3] * 2, out_channels=channels[3], kernel_size=1,
									   stride=1, padding=0)
		self.middle_layer_3 = MiddleLayer(in_channel=channels[3], out_channel=channels[4])
		self.reduce_conv_3 = nn.Conv2d(in_channels=channels[4] * 2, out_channels=channels[4], kernel_size=1,
									   stride=1, padding=0)

	def forward(self, x):
		x1 = self.input_layer.forward(x)
		x1_r = self.reduce_conv_input.forward(x1)
		x2 = self.middle_layer_1.forward(x1_r)
		x2_r = self.reduce_conv_1.forward(x2)
		x3 = self.middle_layer_2.forward(x2_r)
		x3_r = self.reduce_conv_2.forward(x3)
		x4 = self.middle_layer_3.forward(x3_r)
		x4_r = self.reduce_conv_3.forward(x4)
		return x4_r


class Encoder(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.encoder_aux = EncoderAux(channels)

	def forward(self, x):
		x_w = x
		z_w = self.encoder_aux.forward(x_w)
		x_h = torch.rot90(x, 1, (2, 3))
		z_h_a = self.encoder_aux.forward(x_h)
		z_h = torch.rot90(z_h_a, -1, (2, 3))
		return torch.concat([z_w, z_h], 1)


class Decoder(nn.Module):
	def __init__(self, channels=None, last_activation=nn.Tanh()) -> None:
		super().__init__()
		layers = 4
		upscale_ratio = 2
		AppLog.info(
				f'Layers = {layers}, channels = {channels}')

		sequence = nn.Sequential()
		for layer in range(layers):
			ch_in = channels[layer]
			ch_out = channels[layer + 1]

			upsample_layer = nn.Upsample(scale_factor=upscale_ratio, mode='bicubic')
			sequence.append(upsample_layer)
			conv_layer_1, conv_layer_2 = generate_separated_kernels(ch_in, ch_out, 5, r=10 / 25, add_padding=True)
			sequence.append(conv_layer_1)
			sequence.append(conv_layer_2)
			sequence.append(nn.BatchNorm2d(ch_out))
			if layer < layers - 1:
				activation_layer = nn.Mish()
				sequence.append(activation_layer)
			else:
				activation_layer = last_activation
				sequence.append(activation_layer)
		self.sequence = nn.Sequential(*sequence)

	def forward(self, latent_z):
		return self.sequence.forward(latent_z)
