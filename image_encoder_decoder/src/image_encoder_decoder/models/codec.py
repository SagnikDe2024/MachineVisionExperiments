import torch
from numpy import log2
from torch import Tensor, nn
from torch.nn import ModuleDict
from torch.nn.functional import interpolate
from torchinfo import summary


# from ml_common.common_utils import AppLog


def generate_separated_kernels(input_channel: int, output_channel: int, k_size: int, a: float = (1 / 2), r: float = 0.0,
							   add_padding: bool = True, bias=False, stride: int = 1):
	c_in = input_channel
	c_out = output_channel
	t = c_out / c_in
	k = k_size

	if 0 < r < 1 and t != 1:
		a = log2((t + 1) / (t * r * k)) / log2(1 / t)

	c_intermediate = int(round((c_in ** (1 - a) * c_out ** a), 0))
	# AppLog.info(f'c_in={c_in}, c_intermediate={c_intermediate}, c_out={c_out}')
	# if not (0 <= a <= 1):
	# AppLog.warning(
	# f'Inconsistency in intermediate features: {c_intermediate} ∉ [{c_in},{c_out}]')
	padding = k // 2 if add_padding else 0

	conv_layer_1 = nn.Conv2d(c_in, c_intermediate, (1, k), padding=(0, padding), bias=bias, stride=(1, stride))
	conv_layer_2 = nn.Conv2d(c_intermediate, c_out, (k, 1), padding=(padding, 0), bias=bias, stride=(stride, 1))

	return conv_layer_1, conv_layer_2

class EncoderLayer(nn.Module):
	def __init__(self, input_channels, output_channels, kernel_ratios):
		super().__init__()
		total_ratios = sum(kernel_ratios)
		mod_dic = ModuleDict()
		out_channels = [int(round(output_channels * out_r / total_ratios, 0)) for out_r in kernel_ratios]
		rest = sum(out_channels[1:]) if len(out_channels) > 1 else 0
		out_channels[0] = output_channels - rest

		for i, out_ch in enumerate(out_channels):
			kernel_size = i * 2 + 3
			output_channel = out_ch
			if kernel_size == 3:
				conv_layer = nn.Conv2d(in_channels=input_channels, out_channels=output_channel,
									   kernel_size=kernel_size,
									   padding=1, bias=False)
				mod_dic[f'{kernel_size}'] = conv_layer

				continue
			conv_1, conv_2 = generate_separated_kernels(input_channels, output_channel, kernel_size, r=3 / kernel_size,
														add_padding=True)
			seq = nn.Sequential(conv_1, conv_2)
			mod_dic[f'{kernel_size}'] = seq

		self.conv_layers = mod_dic
		self.norm = nn.BatchNorm2d(output_channels)
		self.activation = nn.Mish()



	def forward(self, x):



		convs = []
		for conv_layer in self.conv_layers.values():
			conv : Tensor = conv_layer.forward(x)
			convs.append(conv)

		concat_res = torch.cat(convs, dim=1)

		normed_res = self.norm(concat_res)

		return normed_res




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


# def decoder_upscale_module(input_tensor, kernel_size):
# 	upsample_add = kernel_size - 1
#
# 	interpolate(input_tensor,siz)

class DecoderLayer(nn.Module):
	def __init__(self, input_channels, output_channels, kernel_ratios):
		super().__init__()
		total_ratios = sum(kernel_ratios)
		mod_dic = ModuleDict()
		out_channels = [int(round(output_channels * out_r / total_ratios, 0)) for out_r in kernel_ratios]
		rest = sum(out_channels[1:]) if len(out_channels) > 1 else 0
		out_channels[0] = output_channels - rest

		for i, out_ch in enumerate(out_channels):
			kernel_size = i * 2 + 3
			output_channel = out_ch
			if kernel_size == 3:
				conv_layer = nn.Conv2d(in_channels=input_channels, out_channels=output_channel,
									   kernel_size=kernel_size,
									   padding=1, bias=False)
				mod_dic[f'{kernel_size}'] = conv_layer

				continue
			conv_1, conv_2 = generate_separated_kernels(input_channels, output_channel, kernel_size, r=3 / kernel_size,
														add_padding=True)
			seq = nn.Sequential(conv_1, conv_2)
			mod_dic[f'{kernel_size}'] = seq

		self.conv_layers = mod_dic
		self.norm = nn.BatchNorm2d(output_channels)

		self.h = 128
		self.w = 128

	def set_h_w(self, h, w):
		self.h = h
		self.w = w

	def forward(self, x):

		upsampled = interpolate(x, size=(self.h, self.w), mode='bicubic')

		convs = []
		for conv_layer in self.conv_layers.values():
			conv : Tensor = conv_layer.forward(upsampled)
			convs.append(conv)

		concat_res = torch.cat(convs, dim=1)

		normed_res = self.norm(concat_res)

		return normed_res


class Decoder(nn.Module):
	def __init__(self, in_channels, out_channels) -> None:
		super().__init__()

		layers = 6
		self.layers = layers

		channel_r = (out_channels / in_channels) ** (1 / (layers - 1))
		channels = [int(round(in_channels * channel_r ** x, 0)) for x in range(layers)]
		channels = [*channels, 3]

		# AppLog.info(
		# 		f'Layers = {layers}, channels = {channels}')

		kernel_channel_ratios = [1, 3 / 5, 3 / 7]

		decoder_layers = ModuleDict()
		activation_layers = ModuleDict()
		for layer in range(layers - 1):
			ch_in = channels[layer]
			ch_out = channels[layer + 1]

			dec_layer = DecoderLayer(ch_in, ch_out, kernel_channel_ratios)
			decoder_layers[f'{layer}'] = dec_layer
			activation_layers[f'{layer}'] = nn.Mish()
		last_layer = DecoderLayer(channels[-2], channels[-1], [1])
		decoder_layers['last'] = last_layer
		activation_layers['last'] = nn.Tanh()
		self.decoder_layers = decoder_layers
		self.activation_layers = activation_layers

		self.h = 128
		self.w = 128

	def set(self, h, w):
		self.h = h
		self.w = w

	def forward(self, latent_z: torch.Tensor):
		_, _, z_h, z_w = latent_z.shape
		h_r = self.h / z_h
		w_r = self.w / z_w
		l = self.layers
		h_s = [int(round(z_h * h_r ** ((r + 1) / l), 0)) for r in range(l)]
		w_s = [int(round(z_w * w_r ** ((r + 1) / l), 0)) for r in range(l)]

		z_m = latent_z
		for dec_l, act_l, h, w in zip(self.decoder_layers.values(), self.activation_layers.values(), h_s, w_s):
			dec_l.set_h_w(h, w)
			z_m = dec_l.forward(z_m)
			z_m = act_l.forward(z_m)

		return z_m


if __name__ == '__main__':
	dec = Decoder(in_channels=288, out_channels=48)
	# decl = DecoderLayer(64, 32, [1, 3 / 5, 3 / 7])
	# for layer_name, params in decl.named_parameters():
	# 	print(layer_name, params.shape)
	dec.set(256, 256)
	for layer_name, params in dec.named_parameters():
		print(layer_name, params.shape)
	summary(dec, input_size=(1, 288, 16, 16))
# AppLog.shut_down()
