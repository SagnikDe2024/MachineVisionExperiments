from math import log2
from typing import Any, Tuple

import torch
from torch import nn
from torchinfo import summary

from src.common.common_utils import CNNUtils

"""
UNet for Image Segmentation with Parameter Reduction Techniques

This module implements a UNet architecture for image segmentation with options to reduce
the number of parameters. Two main techniques are implemented:

1. Depthwise Separable Convolutions:
   - Replaces standard convolutions with depthwise separable convolutions
   - Reduces parameters by separating spatial and channel-wise convolutions
   - Can reduce parameters by ~8-9x for a 3x3 convolution

2. Width Multiplier:
   - Scales the number of channels throughout the network by a factor
   - A width_multiplier of 0.5 reduces parameters by ~4x
   - Maintains minimum channel count to ensure model functionality

Usage:
    # Create a UNet with default parameters
    model = SegmentationUnet(channels, num_segments)

    # Create a UNet with depthwise separable convolutions
    model = SegmentationUnet(channels, num_segments, use_separable_conv=True)

    # Create a UNet with reduced width (0.5x channels)
    model = SegmentationUnet(channels, num_segments, width_multiplier=0.5)

    # Create a UNet with both optimizations
    model = SegmentationUnet(channels, num_segments, width_multiplier=0.5, use_separable_conv=True)
"""


def calculate_depth_from_image_dimensions(image_dimensions: Tuple[int, int]) -> list[Any]:
	max_dimension = max(image_dimensions)
	min_dimension = min(image_dimensions)

	layers_and_kernels = []
	find_k = lambda cl, dim: (2 * (2 ** cl + dim - 2)) / (3 * 2 ** cl - 4)

	k_set = set()
	for l in range(4, 14):
		k_raw = find_k(l, max_dimension)
		k = round(k_raw / 2) * 2 + 1
		if k in k_set:
			continue
		k_set.add(k)
		layers_and_kernels.append({'layer': l, 'k': k})
	layers_and_kernels = sorted(layers_and_kernels, key=lambda x: x['k'])
	min_l = layers_and_kernels[0]['layer']
	max_l = layers_and_kernels[-1]['layer']
	mean_l = round((min_l + max_l) / 2)
	layers_and_kernels = list(filter(lambda x: x['layer'] >= mean_l, layers_and_kernels))

	print('All the layers and kernels: ')
	for l_and_k in layers_and_kernels:
		print(f'Layer {l_and_k["layer"]}, kernel size {l_and_k["k"]}')

	first_cnn_layer = 48

	max_param_count = image_dimensions[0] * image_dimensions[1]
	max_parameters = max_param_count

	plausible_params = []

	for l_and_k in layers_and_kernels:
		k = l_and_k['k']
		l = l_and_k['layer']
		for new_ch_ratio in range(l, 0, -1):

			max_final_channels = 2 ** new_ch_ratio * 48
			channel_ratio, all_channels, rest_parameters, all_parameters = get_channels_ratios(first_cnn_layer,
																							   max_final_channels,
																							   l, k)
			if k > 3:
				sep_kernel_min_ratio_rest = max((channel_ratio + 1) / (k * channel_ratio), 9 / (k **
																								2))
				total_params_rest = CNNUtils.calculate_total_cnn_params((k, sep_kernel_min_ratio_rest),
																		all_channels[1:])
				sep_kernel_min_ratio_first = max((16 + 1) / (k * 16), 9 / (k ** 2))
				total_params_first = CNNUtils.calculate_total_cnn_params((k, sep_kernel_min_ratio_first),
																		 [3, first_cnn_layer])
				total_params = total_params_first + total_params_rest
				all_parameters = total_params

			if all_parameters < max_param_count:
				print(f'New baseline, kernel = {k}, cnn_layers = {l}')
				print(f'Calculated CNN channels {all_channels}, ratio = {channel_ratio:.3f}')
				print(f'Maximum parameters for {l} layers is {all_parameters}')
				max_parameters = all_parameters
				plausible_params.append((k, l, max_final_channels, all_parameters / (2 ** 20)))
			# break

	return plausible_params


def get_channels_ratios(first_cnn_layer, max_final_channels, layers_required, kernel_size=3):
	channel_ratio = (max_final_channels / first_cnn_layer) ** (1 / layers_required)
	channels = [round(first_cnn_layer * channel_ratio ** x) for x in range(layers_required + 1)]
	all_channels = [3, *channels]
	max_parameters = CNNUtils.calculate_total_cnn_params(kernel_size, all_channels)
	rest_parameters = CNNUtils.calculate_total_cnn_params(kernel_size, channels)
	return channel_ratio, all_channels, rest_parameters, max_parameters


class DepthwiseSeparableConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
		super().__init__()
		self.depthwise = nn.Conv2d(
				in_channels,
				in_channels,
				kernel_size=kernel_size,
				padding=padding,
				stride=stride,
				groups=in_channels
		)
		self.pointwise = nn.Conv2d(
				in_channels,
				out_channels,
				kernel_size=1
		)

	def forward(self, x):
		x = self.depthwise(x)
		x = self.pointwise(x)
		return x


class SegmentationUnet(nn.Module):
	def __init__(self, channels, segments, width_multiplier=1.0, use_separable_conv=True):
		super().__init__()
		# Apply width multiplier to reduce channels
		if width_multiplier != 1.0:
			channels = [3] + [max(8, int(ch * width_multiplier)) for ch in channels[1:]]

		down_sample = nn.ModuleDict()
		input_channels = channels[:-1]
		output_channels = channels[1:]
		total_layers = len(output_channels)

		self.middle_activation = nn.Mish()

		for i, ch in enumerate(zip(input_channels, output_channels)):
			inp_ch, out_ch = ch
			if use_separable_conv and i > 0:  # Use regular conv for first layer (from RGB)
				conv = DepthwiseSeparableConv(inp_ch, out_ch, kernel_size=3, padding=1)
			else:
				conv = nn.Conv2d(inp_ch, out_ch, kernel_size=3, padding=1)
			norm = nn.GroupNorm(1, out_ch)
			act = nn.Mish()
			pool = nn.MaxPool2d(2)
			down_sample[f'conv_down_{i}'] = nn.Sequential(conv, norm, act)
			down_sample[f'pool_{i}'] = pool

		up_sample = nn.ModuleDict()
		up_sample_input_channels_raw = output_channels[::-1]
		up_sample_output_channels = [*up_sample_input_channels_raw[1:], segments]

		# Double the input channels for skip connections
		up_sample_input_channels = list(map(lambda x: 2 * x, up_sample_input_channels_raw))

		print(f'Up sample input channels: {up_sample_input_channels}')
		print(f'Up sample output channels: {up_sample_output_channels}')

		for i, (inp_ch, out_ch) in enumerate(zip(up_sample_input_channels, up_sample_output_channels)):
			up_pool = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
			if use_separable_conv:
				conv = DepthwiseSeparableConv(inp_ch, out_ch, kernel_size=3, padding=1)
			else:
				conv = nn.Conv2d(inp_ch, out_ch, kernel_size=3, padding=1)
			norm = nn.GroupNorm(1, out_ch)
			depth = total_layers - i - 1
			act = nn.Mish() if depth > 0 else nn.Softmax2d()
			up_sample[f'up_pool_{depth}'] = up_pool
			up_sample[f'conv_up_{depth}'] = nn.Sequential(conv, norm, act)

		self.down_sample = down_sample
		self.up_sample = up_sample

	def forward(self, x):
		downsampled = []
		for k, layer in self.down_sample.items():
			if 'conv_down' in k:
				conved = layer(x)
				downsampled.append(conved)
				x = conved
			else:
				x = layer(x)
				print(f'Downsampled shape: {x.shape}')

		x = self.middle_activation(x)
		print(f'Middle shape: {x.shape}')
		for k, layer in self.up_sample.items():
			if 'up_pool' in k:
				x = layer(x)
			else:
				x = layer(torch.cat([x, downsampled.pop()], dim=1))
			print(f'Upsampled shape: {x.shape}')
		return x


if __name__ == '__main__':
	h, w = 512, 1024
	total_info = h * w
	max_dim = max(h, w)
	min_dim = min(h, w)
	max_range = round(log2(min_dim))
	layers = -1
	for k in range(4, max_range):
		cov = CNNUtils.calculate_coverage(3, k, 2)
		layers = k
		if cov >= min_dim:
			break
	print(f'Max layers = {layers}')
	start_ch = 32
	end_ch = 128
	layers = (layers + 4) // 2

	channel_ratio = (end_ch / start_ch) ** (1 / layers)
	channels = [round(start_ch * channel_ratio ** l / 12) * 12 for l in range(layers + 1)]
	all_channels = [3, *channels]

	# Example 1: Original UNet (baseline)
	print("\n=== Original UNet (Baseline) ===")
	seg_original = SegmentationUnet(all_channels, 5, width_multiplier=1.0, use_separable_conv=False)
	summary(seg_original, input_size=(1, 3, 512, 1024))

	# Example 2: UNet with depthwise separable convolutions
	print("\n=== UNet with Depthwise Separable Convolutions ===")
	seg_separable = SegmentationUnet(all_channels, 5, width_multiplier=1.0, use_separable_conv=True)
	summary(seg_separable, input_size=(1, 3, 512, 1024))

	# Example 3: UNet with reduced width (0.5x channels)
	print("\n=== UNet with Reduced Width (0.5x) ===")
	seg_reduced_width = SegmentationUnet(all_channels, 5, width_multiplier=0.5, use_separable_conv=False)
	summary(seg_reduced_width, input_size=(1, 3, 512, 1024))

	# Example 4: UNet with both optimizations
	print("\n=== UNet with Both Optimizations ===")
	seg_optimized = SegmentationUnet(all_channels, 5, width_multiplier=0.5, use_separable_conv=True)
	summary(seg_optimized, input_size=(1, 3, 512, 1024))

	print(f'Max param acceptable = {total_info}')

# params = calculate_depth_from_image_dimensions((5120, 2160))
# params = sorted(params, key=lambda x: (x[0],-x[2]))
# for p in params:
# 	print(p)
