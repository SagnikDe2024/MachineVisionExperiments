import torch
from torch import Tensor, nn
from torch.nn import ModuleDict
from torch.nn.functional import interpolate
from torchinfo import summary

from src.common.common_utils import Ratio, generate_separated_kernels


class EncoderLayer3Conv(nn.Module):
	def __init__(self, input_channels, output_channels, kernels_and_ratios, downsample=0.5):
		super().__init__()
		kernels, kernel_ratios = zip(*kernels_and_ratios)
		total_ratios = sum(kernel_ratios)
		mod_dic = ModuleDict()
		out_channels = [int(round(output_channels * out_r / total_ratios, 0)) for out_r in kernel_ratios]
		rest = sum(out_channels[1:]) if len(out_channels) > 1 else 0
		out_channels[0] = output_channels - rest




class Encoder(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.encoder_aux = Encoder(channels)

	def forward(self, x):
		x_w = x
		z_w = self.encoder_aux.forward(x_w)
		x_h = torch.rot90(x, 1, (2, 3))
		z_h_a = self.encoder_aux.forward(x_h)
		z_h = torch.rot90(z_h_a, -1, (2, 3))
		return torch.concat([z_w, z_h], 1)



class EncoderLayer(nn.Module):
	def __init__(self, input_channels: int, output_channels: int, kernels_and_ratios, downsample: float = 0.5):
		super().__init__()
		kernels, kernel_ratios = zip(*kernels_and_ratios)
		total_ratios = sum(kernel_ratios)
		mod_dic = ModuleDict()
		out_channels = [int(round(output_channels * out_r / total_ratios, 0)) for out_r in kernel_ratios]
		rest = sum(out_channels[1:]) if len(out_channels) > 1 else 0
		out_channels[0] = output_channels - rest

		for i, out_ch in enumerate(out_channels):
			kernel_size = kernels[i]
			output_channel = out_ch
			if kernel_size == 3:
				conv_layer = nn.Conv2d(in_channels=input_channels, out_channels=output_channel,
									   kernel_size=kernel_size,
									   padding=1, bias=False)
				mod_dic[f'{kernel_size}'] = conv_layer
				continue
			conv_1, conv_2 = generate_separated_kernels(input_channels, output_channel, kernel_size,
														Ratio(3 / kernel_size), add_padding=True)
			seq = nn.Sequential(conv_1, conv_2)
			mod_dic[f'{kernel_size}'] = seq

		self.conv_layers = mod_dic
		self.norm = nn.BatchNorm2d(output_channels)
		self.activation = nn.Mish()
		self.pooling = nn.FractionalMaxPool2d(2, output_channels, downsample)

	def forward(self, x: Tensor):
		convs = []
		for conv_layer in self.conv_layers.values():
			conv: Tensor = conv_layer.forward(x)
			convs.append(conv)
		concat_res = torch.cat(convs, dim=1)
		normed_res = self.norm(concat_res)
		active_res = self.activation(normed_res)
		# Use resnet style skip connection
		pooled_res = self.pooling(active_res + normed_res)
		return pooled_res


# def decoder_upscale_module(input_tensor, kernel_size):
# 	upsample_add = kernel_size - 1
#
# 	interpolate(input_tensor,siz)

class DecoderLayer(nn.Module):

	def __init__(self, input_channels, output_channels, kernels_and_ratios, upscale=None):
		super().__init__()
		kernels, kernel_ratios = zip(*kernels_and_ratios)


		total_ratios = sum(kernel_ratios)
		mod_dic = ModuleDict()
		out_channels = [int(round(output_channels * out_r / total_ratios, 0)) for out_r in kernel_ratios]
		rest = sum(out_channels[1:]) if len(out_channels) > 1 else 0
		out_channels[0] = output_channels - rest

		for i, out_ch in enumerate(out_channels):
			kernel_size = kernels[i]

			output_channel = out_ch
			if kernel_size == 3:
				conv_layer = nn.Conv2d(in_channels=input_channels, out_channels=output_channel,
									   kernel_size=kernel_size,
									   padding=1, bias=False)
				mod_dic[f'{kernel_size}'] = conv_layer
				continue
			conv_1, conv_2 = generate_separated_kernels(input_channels, output_channel, kernel_size,
														Ratio(3 / kernel_size), add_padding=True)

			seq = nn.Sequential(conv_1, conv_2)
			if upscale is not None:
				self.upscale = nn.Upsample(scale_factor=upscale, mode='bilinear')
			mod_dic[f'{kernel_size}'] = seq

		self.conv_layers = mod_dic
		self.norm = nn.InstanceNorm2d(output_channels)


		self.h = 128
		self.w = 128

	def set_h_w(self, h, w):
		self.h = h
		self.w = w


	def forward(self, x: Tensor):

		if self.upscale is not None:
			upsampled = self.upscale(x)
		else:
			upsampled: Tensor = interpolate(x, size=(self.h, self.w), mode='bicubic')

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
