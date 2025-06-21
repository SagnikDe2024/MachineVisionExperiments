import torch
from torch import Tensor, nn
from torch.nn import ModuleDict
from torch.nn.functional import interpolate

from src.common.common_utils import ParamA, generate_separated_kernels


# class L1BatchNorm2D(nn.Module):
# 	def __init__(self, channels):
# 		super().__init__()
# 		self.bnl1_weights = nn.Parameter(torch.ones(channels))
# 		self.bnl1_bias = nn.Parameter(torch.zeros(channels))
# 		self.bnl1_const = math.pi/2
#
# 	def forward(self, x):
#
# 		l1_mean = torch.mean(x, dim=-4, keepdim=True)
# 		diff = x - l1_mean
# 		l1_std = torch.abs(diff).mean(dim=-4, keepdim=True)
# 		x_normed = diff / (self.bnl1_const*l1_std + 1e-8)
# 		x_normed_weights_bias = x_normed * self.bnl1_weights + self.bnl1_bias
# 		return x_normed_weights_bias


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
			                                            ParamA(0), add_padding=True)
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
		self.upscale = None
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
			                                            ParamA(1), add_padding=True)

			seq = nn.Sequential(conv_1, conv_2)
			if upscale is not None:
				self.upscale = nn.Upsample(scale_factor=upscale, mode='bicubic')
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



