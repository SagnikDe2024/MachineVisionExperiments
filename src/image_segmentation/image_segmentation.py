from math import log2
from typing import Any, Tuple

import torch
from torch import nn
from torch.nn.functional import interpolate
from torchinfo import summary
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2.functional import gaussian_noise, resize_image, rotate

from src.common.common_utils import CNNUtils, IntermediateChannel, ParamA, Ratio, generate_separated_kernels, \
	get_diffs

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


def get_single_channel_max_diff(image):
	diff_h, diff_w = get_diffs(image)
	max_diff_h = torch.argmax(torch.max(torch.abs(diff_h)), dim=0)
	single_channel_diff_h = diff_h[max_diff_h]
	max_diff_w = torch.argmax(torch.max(torch.abs(diff_w)), dim=0)
	single_channel_diff_w = diff_w[max_diff_w]
	return torch.unsqueeze(single_channel_diff_h, 0), torch.unsqueeze(single_channel_diff_w, 0)


def prepare_a_single_image(image, res_i):
	image_dict = {}
	for i in range(4):
		angle = i * 90
		img = rotate(image, angle=angle, interpolation=InterpolationMode.BILINEAR, expand=True)

		diff_h, diff_w = get_single_channel_max_diff(img)
		image_dict[f'{res_i}_r_{angle}'] = img, diff_h, diff_w
	flipped = torch.flip(image, dims=[1])
	for i in range(4):
		angle = i * 90
		img_f = rotate(flipped, angle, interpolation=InterpolationMode.BILINEAR, expand=True)
		diff_h_f, diff_w_f = get_single_channel_max_diff(img_f)
		image_dict[f'{res_i}_fr_{angle}'] = img_f, diff_h_f, diff_w_f
	return image_dict


def augment_a_single_image(image, times=5):
	(c, h, w) = image.shape
	min_dim = min(h, w)
	max_dim = max(h, w)
	lowend_ratio_log = log2(64 / min_dim)
	upperend_ratio_log = log2(1024 / max_dim)
	log_divs = (upperend_ratio_log - lowend_ratio_log) / times
	prepared_images = {}
	for log_index in range(times + 1):
		log_ratio = log_divs * log_index + lowend_ratio_log
		ratio = 2 ** log_ratio
		new_h = round(h * ratio)
		new_w = round(w * ratio)
		resized = resize_image(image, [new_h, new_w], antialias=True)
		image_dict = prepare_a_single_image(resized, log_index)
		prepared_images.update(image_dict)
	return prepared_images


def augment_with_noise(image, times=32, max_size=1024):
	(c, h, w) = image.shape
	max_dim = max(h, w)
	upperend_ratio = (max_size / max_dim)
	rh, rw = round(h * upperend_ratio), round(w * upperend_ratio)
	resized = resize_image(image, [rh, rw], antialias=True)
	images_made = [resized]
	for t in range(times):
		noised = gaussian_noise(resized, 0.01)
		images_made.append(noised)
	return images_made


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


def get_separated_conv_kernel(in_channels, out_channels, inter_ch, kernel_size=3, switch=False):
	conv1, conv2 = generate_separated_kernels(in_channels, out_channels, kernel_size, inter_ch=inter_ch, switch=switch)
	return nn.Sequential(conv1, conv2)


class SegmentationUnet(nn.Module):
	def __init__(self, down_sample_channels, upsample_channels):
		super().__init__()
		self.model_params = {'down_sample_channels': down_sample_channels, 'upsample_channels': upsample_channels}
		down_sample = nn.ModuleDict()
		input_channels = down_sample_channels[:-1]
		output_channels = down_sample_channels[1:]
		total_layers = len(upsample_channels)

		for i, ch in enumerate(zip(input_channels, output_channels)):
			inp_ch, out_ch = ch
			if i > 0:  # Use regular conv for first layer (from RGB)
				conv = get_separated_conv_kernel(inp_ch, out_ch, inter_ch=ParamA(0), kernel_size=5, switch=False)
			else:
				conv = nn.Conv2d(inp_ch, out_ch, kernel_size=3, padding=1, bias=False)
			norm = nn.GroupNorm(1, out_ch)
			act = nn.Mish()
			pool = nn.MaxPool2d(2)
			down_sample[f'conv_down_{i}'] = nn.Sequential(conv, norm, act)
			down_sample[f'pool_{i}'] = pool

		up_sample = nn.ModuleDict()
		# Double the input channels for skip connections
		print(f'Up sample channels: {upsample_channels}')

		upsample_channels_inp_unet = upsample_channels[:-1]

		for i, (inp_ch, out_ch) in enumerate(upsample_channels_inp_unet):
			conv = get_separated_conv_kernel(inp_ch, out_ch, inter_ch=IntermediateChannel(intermediate_channel=out_ch),
											 kernel_size=3, switch=True)

			norm = nn.GroupNorm(1, out_ch)
			depth = total_layers - i - 1
			act = nn.Mish() if depth > 0 else nn.Softmax2d()

			up_sample[f'conv_up_{depth}'] = nn.Sequential(conv, norm, act)

		self.down_sample = down_sample
		self.up_sample = up_sample

		final_conv = get_separated_conv_kernel(upsample_channels[-1][0], upsample_channels[-1][1],
											   inter_ch=Ratio(2 / 3), kernel_size=3, switch=True)
		final_norm = nn.GroupNorm(1, upsample_channels[-1][-1])
		final_act = nn.Softmax2d()
		self.final_conv = nn.Sequential(final_conv, final_norm, final_act)

	def forward(self, x):
		downsampled = []

		for k, layer in self.down_sample.items():
			if 'conv_down' in k:
				conved = layer.forward(x)
				downsampled.append(conved)
				x = conved
			else:
				x = layer.forward(x)

		up_conv_i = 0

		for m in self.up_sample.items():
			k, layer = m

			if up_conv_i == 0:
				conved_up = layer.forward(x)
			else:
				skip_tensor = downsampled.pop()
				conved_up = layer.forward(torch.cat([x, skip_tensor], dim=1))
				up_conv_i += 1
			x = conved_up

			[_, _, h, w] = downsampled[-1].shape
			x = interpolate(x, size=(h, w), mode='bicubic', align_corners=False)

		x = self.final_conv.forward(torch.cat([x, downsampled.pop()], dim=1))

		return x


def create_unet_layers(top_level_down_channels, bottom_level_down_channels, top_level_up_channels,
					   segments, layers):
	channel_ratio_down = (bottom_level_down_channels / top_level_down_channels) ** (1 / (layers - 1))
	channels_down = [round(top_level_down_channels * channel_ratio_down ** l) for l in range(layers)]

	channel_ratio_up = (bottom_level_down_channels / top_level_up_channels) ** (1 / layers)
	channels_up = [round(top_level_up_channels * channel_ratio_up ** l) for l in range(layers + 1)]
	channels_up_input_uplayers = list(map(lambda x: x[0] + x[1], zip(channels_down, channels_up[:-1])))
	channels_up_input = [*channels_up_input_uplayers, channels_up[-1]]

	channels_down = [3, *channels_down]
	channels_up_output = [segments, *channels_up]
	channels_up = list(zip(channels_up_input, channels_up_output))

	return channels_down, channels_up


def get_unet():
	ch_d, ch_up = create_unet_layers(32, 192, 30, 2, 6)
	down_sample_channels = ch_d
	upsample_channels = ch_up[::-1]
	unet = SegmentationUnet(down_sample_channels, upsample_channels)
	return unet


if __name__ == '__main__':
	ch_d, ch_up = create_unet_layers(32, 192, 30, 5, 6)
	print(ch_d)
	print(ch_up)

	down_sample_channels = ch_d
	upsample_channels = ch_up[::-1]

	unet = SegmentationUnet(down_sample_channels, upsample_channels)
	# print(unet)
	summary(unet, input_size=(1, 3, 512, 1024))

# h, w = 512, 1024
# total_info = h * w
# max_dim = max(h, w)
# min_dim = min(h, w)
# max_range = round(log2(min_dim))
# layers = -1
# for k in range(4, max_range):
# 	cov = CNNUtils.calculate_coverage(3, k, 2)
# 	layers = k
# 	if cov >= min_dim:
# 		break
# print(f'Max layers = {layers}')
# starting_channels_d = 32
# end_ch = 128
# layers = (layers + 4) // 2
#
# channel_ratio = (end_ch / starting_channels_d) ** (1 / layers)
# channels = [round(starting_channels_d * channel_ratio ** l / 12) * 12 for l in range(layers + 1)]
# all_channels = [3, *channels]
#
# # Example 1: Original UNet (baseline)
# print("\n=== Original UNet (Baseline) ===")
# seg_original = SegmentationUnet(all_channels, 5, width_multiplier=1.0, use_separable_conv=False)
# summary(seg_original, input_size=(1, 3, 512, 1024))
#
# # Example 2: UNet with depthwise separable convolutions
# print("\n=== UNet with Depthwise Separable Convolutions ===")
# seg_separable = SegmentationUnet(all_channels, 5, width_multiplier=1.0, use_separable_conv=True)
# summary(seg_separable, input_size=(1, 3, 512, 1024))
#
# # Example 3: UNet with reduced width (0.5x channels)
# print("\n=== UNet with Reduced Width (0.5x) ===")
# seg_reduced_width = SegmentationUnet(all_channels, 5, width_multiplier=0.5, use_separable_conv=False)
# summary(seg_reduced_width, input_size=(1, 3, 512, 1024))
#
# # Example 4: UNet with both optimizations
# print("\n=== UNet with Both Optimizations ===")
# seg_optimized = SegmentationUnet(all_channels, 5, width_multiplier=0.5, use_separable_conv=True)
# summary(seg_optimized, input_size=(1, 3, 512, 1024))
#
# print(f'Max param acceptable = {total_info}')

# params = calculate_depth_from_image_dimensions((5120, 2160))
# params = sorted(params, key=lambda x: (x[0],-x[2]))
# for p in params:
# 	print(p)
