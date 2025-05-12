from math import log2
from typing import List

from torch import nn
from torchinfo import summary

from src.common.common_utils import AppLog


# This will classify the CIFAR-10 model into the classes for now.

class Classifier(nn.Module):
	def __init__(self, fcn_layers: List[int], starting_size: int, final_size: int, starting_channels: int,
				 final_channels: int, cnn_layers: int) -> None:
		super().__init__()
		self.model_params = {'fcn_layers': fcn_layers, 'starting_size': starting_size, 'final_size': final_size,
							 'starting_channels': starting_channels,
							 'final_channels'   : final_channels, 'cnn_layers': cnn_layers}

		channel_ratio = (final_channels / starting_channels) ** (1 / (cnn_layers - 1))
		AppLog.info(f"Classifier channel upscale ratio: {channel_ratio}")

		channels_rest = [round(starting_channels * channel_ratio ** l) for l in range(cnn_layers)]

		channels = [3, *channels_rest]
		kernels = [3 for _ in range(cnn_layers)]
		encoder = Encoder(starting_size, final_size, kernels, channels)
		first_fcn_layer = final_size ** 2 * final_channels
		fcn_layers = [first_fcn_layer, *fcn_layers]
		AppLog.info(f"FCN layers: {fcn_layers}")

		self.encoder = encoder
		sequence = nn.Sequential()
		# sequence.append(nn.Mish())

		# Flatten the result from the encoder first
		sequence.append(nn.Flatten())

		for layer_fcn in range(len(fcn_layers) - 2):
			# Apply an activation
			act = nn.Mish()
			lin = nn.Linear(fcn_layers[layer_fcn], fcn_layers[layer_fcn + 1])
			sequence.append(lin)
			sequence.append(act)
		sequence.append(nn.Linear(fcn_layers[-2], fcn_layers[-1]))

		self.sequence = nn.Sequential(*sequence)
		self.normalized = nn.Softmax(dim=1)

	def forward(self, x):
		features = self.encoder.forward(x)
		raw_probability_values = self.sequence(features)
		probabilities = self.normalized(raw_probability_values)
		# Normalize when using the model.
		return raw_probability_values, probabilities


class EncoderCNNBlock(nn.Module):
	def __init__(self, ch_in, ch_out) -> None:
		super().__init__()
		self.conv = nn.Conv2d(ch_in, ch_out, 3, stride=1, padding=1, bias=False)
		self.act = nn.Mish()
		self.bn = nn.BatchNorm2d(ch_out)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		fx = self.act(x)
		return x + fx


class Encoder(nn.Module):
	def __init__(self, input_size, output_size, kernel_sizes=None, channels=None) -> None:
		super().__init__()

		layers = len(kernel_sizes)
		downscale_ratio = (output_size / input_size) ** (1 / layers)
		sequence = []
		downsampled_sizes = [int(round(input_size * downscale_ratio ** (layer + 1), 0)) for layer in range(layers)]
		AppLog.info(
				f'Layers = {layers}, downsampled_sizes = {downsampled_sizes}, channels = {channels}')

		for layer in range(layers):
			ch_in, ch_out = channels[layer], channels[layer + 1]

			conv_batch_activation = EncoderCNNBlock(ch_in, ch_out)

			pooling_layer = nn.FractionalMaxPool2d(2, output_size=downsampled_sizes[layer])

			sequence.append(conv_batch_activation)
			sequence.append(pooling_layer)
		self.sequence = nn.Sequential(*sequence)

	@classmethod
	def single_kernel_encode(cls, input_size, output_size, kernel_sizes, inp_out_channels):
		layers = len(kernel_sizes)

		if len(inp_out_channels) == 2 and layers > 1:
			channels = channel_kernel_compute(inp_out_channels, layers)
		elif len(inp_out_channels) == 3 and layers > 2:
			channels_later = channel_kernel_compute(inp_out_channels[1:], layers - 1)
			channels = [inp_out_channels[0], *channels_later]
		else:
			channels = [*inp_out_channels]
		print(f'Encoder channels and kernels: {channels},{kernel_sizes}')
		enc = Encoder(input_size, output_size, kernel_sizes, channels)
		return enc

	def forward(self, input_x):
		return self.sequence.forward(input_x)


# The function will make a 1×k kernel followed by a k×1 kernel with c_in^(1-a)×c_out^a intermediate channels.
# By default, c_intermediate is geometric mean of c_in and c_out i.e. a = 1/2.
# The other parameter that can be used is 'r' which is the ratio of parameters of an original k×k kernel.
# Since a k×k kernel has (k×k×c_in + 1)×c_out parameters, the number total number of parameters
# of the new two kernels will be roughly r×(k×k×c_in + 1)×c_out parameters.
# The 'a' will be calculated automatically. It will warn if the calculated 'a' falls outside [0,1].

def generate_separated_kernels(k_size: int, input_channel: int, output_channel: int, a: float = (1 / 2), r: float = 0.0,
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

	conv_layer_1 = nn.Conv2d(c_in, c_intermediate, (k, 1), padding=(padding, 0), bias=bias, stride=stride)
	conv_layer_2 = nn.Conv2d(c_intermediate, c_out, (1, k), padding=(0, padding), bias=bias, stride=stride)
	return conv_layer_1, conv_layer_2


def channel_kernel_compute(inp_out_channels: List[int], layers: int):
	in_channel = inp_out_channels[0]
	out_channel = inp_out_channels[1]
	ratio = (out_channel / in_channel) ** (1 / layers)
	channels = [round(in_channel * ratio ** x) for x in range(layers + 1)]
	return channels


if __name__ == '__main__':
	classifier = Classifier([256, 128, 10], 32, 2, 54, 192, 7)

	AppLog.info(f'{summary(classifier, input_size=(128, 3, 32, 32))}')
