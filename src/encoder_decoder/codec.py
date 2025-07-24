from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import ModuleDict
from torch.nn.functional import interpolate

from src.common.common_utils import AppLog, IntermediateChannel, ParamA, generate_separated_kernels


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
	def __init__(self, input_ch, mid_ch, output_ch, stack: list[Tuple[int, int]] | Tuple[int, list[int]]):
		super().__init__()
		self.cnn_stack = CodecMultiKernelStack(input_ch, mid_ch, output_ch, stack)
		self.poolingAvg = nn.AvgPool2d(kernel_size=2)
		self.poolingMax = nn.MaxPool2d(kernel_size=2)
		self.poolSelector = PoolSelector(127)

	def forward(self, x: Tensor):
		x_res = self.cnn_stack(x)
		pooledAvg = self.poolingAvg(x_res)
		pooledMax = self.poolingMax(x_res)
		selected = self.poolSelector(x_res)
		selected_dim = selected.view(selected.shape[0], selected.shape[1], 1, 1)
		pooled = pooledAvg * selected_dim + pooledMax * (1 - selected_dim)
		return pooled




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
			conv_1, conv_2 = generate_separated_kernels(input_channels, output_channel, kernel_size, ParamA(1),
			                                            add_padding=True)

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
			conv: Tensor = conv_layer.forward(upsampled)
			convs.append(conv)

		concat_res = torch.cat(convs, dim=1)

		normed_res = self.norm(concat_res)

		return normed_res


def create_sep_kernels(input_channels, output_channels, kernel_size):
	min_channels = min(input_channels, output_channels)
	padding = kernel_size // 2
	conv1 = nn.Conv2d(in_channels=input_channels, out_channels=min_channels, kernel_size=(1, kernel_size),
	                  padding=(0, padding), bias=False)
	conv2 = nn.Conv2d(in_channels=min_channels, out_channels=output_channels, kernel_size=(kernel_size, 1),
	                  padding=(padding, 0), bias=False)
	return conv1, conv2


class EncoderLayer1stParamAux(nn.Module):
	def __init__(self, in_ch, out_ch, out_ch_end, k_size):
		super().__init__()
		k1_1, k1_2 = generate_separated_kernels(in_ch, out_ch, k_size, IntermediateChannel(in_ch), stride=2,
		                                        switch=True)
		self.activatedk1 = nn.Sequential(k1_1, k1_2, nn.BatchNorm2d(out_ch), nn.Mish())
		k2_1, k2_2 = generate_separated_kernels(in_ch, out_ch, k_size, IntermediateChannel(in_ch), stride=2,
		                                        switch=False)
		self.activatedk2 = nn.Sequential(k2_1, k2_2, nn.BatchNorm2d(out_ch), nn.Mish())
		self.compress = nn.LazyConv2d(out_channels=out_ch_end, kernel_size=1, padding=0, bias=False)

	def forward(self, x):
		x1 = self.activatedk1(x)
		x2 = self.activatedk2(x)
		x_compressed = self.compress(torch.cat([x1, x2], dim=1))
		return x_compressed


class EncoderLayer1stPart2(nn.Module):
	def __init__(self, output_channels, kernels=1):
		super().__init__()
		input_channels = 3
		out_ch = output_channels
		out_ch_end = round(output_channels / kernels)
		if kernels > 1:
			k = kernels
			out_ch = round(-(9 * (2 * k ** 2 + k * (4 - 3 * out_ch) - 6)) / (6 * k ** 2 + 12 * k + 2 * out_ch + 9))
		print(f'Output channels {out_ch}')
		if kernels == 1:
			self.k3 = nn.Sequential(nn.Conv2d(input_channels, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(
					out_ch),
			                        nn.Mish(), nn.MaxPool2d(2))
		else:
			self.k3 = nn.Sequential(nn.Conv2d(input_channels, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(
					out_ch),
			                        nn.Mish(), nn.MaxPool2d(2), nn.LazyConv2d(out_ch_end, 1, padding=0, bias=False))
		for k_o in range(2, kernels + 1):
			k_size = 2 * k_o + 1
			self.add_module(f'k{k_size}', EncoderLayer1stParamAux(input_channels, out_ch, out_ch_end, k_size))

	def forward(self, x):
		result = [self.k3(x)]
		for k in self.children():
			result.append(k(x))

		return torch.cat(result, dim=1)



class PoolSelector(nn.Module):
	def __init__(self, m_size):
		super().__init__()

		side_size = int(m_size ** (1 / 3))
		self.avg_pool = nn.AdaptiveAvgPool2d(side_size)
		self.conv_down = nn.LazyConv2d(side_size, kernel_size=1, padding=0, bias=True)
		new_size = side_size ** 3
		self.linear_part = nn.Sequential()
		AppLog.info(f'The size is {new_size}')
		while new_size >= 2:
			down_size = new_size // 2
			lin = nn.Linear(new_size, down_size, bias=False)
			norm = nn.BatchNorm1d(down_size)
			self.linear_part.append(lin)
			self.linear_part.append(norm)

			if down_size == 1:
				self.linear_part.append(nn.Sigmoid())
				break
			else:
				self.linear_part.append(nn.Mish())
			new_size = down_size
		self.all_ops = nn.Sequential(self.avg_pool, self.conv_down, nn.Flatten(), self.linear_part)

	def forward(self, x):
		return self.all_ops(x)


class CodecSubBlock(nn.Module):
	def __init__(self, in_ch, mid_ch, out_ch, kernel_size):
		super().__init__()
		self.input_compress = nn.Conv2d(in_ch, mid_ch, kernel_size=1, padding=0,
		                                bias=False) if in_ch != mid_ch else nn.Identity()
		conv1, conv2 = create_sep_kernels(mid_ch, mid_ch, kernel_size)
		self.output_compress = nn.Conv2d(mid_ch, out_ch, kernel_size=1, padding=0)
		self.active_path = nn.Sequential(self.input_compress, conv1, conv2, nn.BatchNorm2d(mid_ch), nn.Mish(),
		                                 self.output_compress)

	def forward(self, x):
		return self.active_path(x)


class CodecMultiKernelBlock(nn.Module):
	def __init__(self, in_channels, middle_ch, out_channels, kernels: list[int]):
		super().__init__()
		n = len(kernels)
		AppLog.info(f'CodecMultiKernelBlock: i={in_channels} m={middle_ch} o={out_channels} ks={kernels} ')
		# This is to ensure that the sum of output channels is exactly the same as the number of output_channels
		out_each_ch = [round(i * out_channels / n) for i in range(0, n + 1)]
		self.passthrough = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
		self.kernels = ModuleDict()
		for i, k_s in enumerate(kernels):
			out_each = out_each_ch[i + 1] - out_each_ch[i]
			self.kernels[f'k{i}'] = CodecSubBlock(in_channels, middle_ch, out_each, k_s)

	def forward(self, x):
		evaluated = []
		passthrough = self.passthrough(x)
		for act in self.kernels.values():
			activated = act(x)
			evaluated.append(activated)
		concat_activated = torch.cat(evaluated, dim=1)
		return concat_activated + passthrough


class CodecMultiKernelStack(nn.Module):
	def __init__(self, input_ch, mid_ch, output_ch, stack: list[Tuple[int, int]] | Tuple[int, list[int]]):
		super().__init__()
		if stack is list[Tuple[int, int]]:
			_stack = [list(range(k[0], k[1] * 2 + k[0], 2)) for k in stack]
		else:
			kst = stack[1]
			stacks = stack[0]
			_stack = [[*kst] for _ in range(stacks)]
		stacks = len(_stack)

		self.kernel_stack = nn.Sequential()
		if stacks == 1:
			enc_block = CodecMultiKernelBlock(input_ch, output_ch, mid_ch, _stack[0])
			self.kernel_stack.append(enc_block)
		else:
			for i, ks in enumerate(_stack):
				if i == 0:
					enc_block = CodecMultiKernelBlock(input_ch, mid_ch, mid_ch, ks)
				elif i == stacks - 1:
					enc_block = CodecMultiKernelBlock(mid_ch, mid_ch, output_ch, ks)
				else:
					enc_block = CodecMultiKernelBlock(mid_ch, mid_ch, mid_ch, ks)
				self.kernel_stack.append(enc_block)
		self.poolingAvg = nn.AvgPool2d(kernel_size=2)
		self.poolingMax = nn.MaxPool2d(kernel_size=2)
		self.poolSelector = PoolSelector(127)

	def forward(self, x):
		x_res = self.kernel_stack(x)
		return x_res
