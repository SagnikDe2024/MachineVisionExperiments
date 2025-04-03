from typing import List

from torch import nn
from torchinfo import summary

from src.models.codec import Encoder
from src.utils.common_utils import AppLog


# This will classify the CIFAR-10 model into the classes for now. Might be helpful for checking the generation
# when it will be used later.

class Classifier(nn.Module):
	def __init__(self, dnn_layers: List[int], starting_size: int, final_size: int, starting_channels: int,
				 final_channels: int, cnn_layers: int) -> None:
		super().__init__()
		self.model_params = {'dnn_layers'       : dnn_layers, 'starting_size': starting_size, 'final_size': final_size,
							 'starting_channels': starting_channels,
							 'final_channels'   : final_channels, 'cnn_layers': cnn_layers}

		channel_ratio = (final_channels / starting_channels) ** (1 / (cnn_layers - 1))
		AppLog.info(f"Classifier channel upscale ratio: {channel_ratio}")

		channels_rest = [round(starting_channels * channel_ratio ** l) for l in range(cnn_layers)]

		channels = [3, *channels_rest]
		kernels = [3 for _ in range(cnn_layers)]
		encoder = Encoder(starting_size, final_size, kernels, channels)
		first_dnn_layer = final_size ** 2 * final_channels
		dnn_layers = [first_dnn_layer, *dnn_layers]
		AppLog.info(f"DNN layers: {dnn_layers}")

		self.encoder = encoder
		sequence = nn.Sequential()
		sequence.append(nn.Mish())

		# Flatten the result from the encoder first
		sequence.append(nn.Flatten())

		for layer_dnn in range(len(dnn_layers) - 2):
			# Apply an activation
			act = nn.Mish()
			lin = nn.Linear(dnn_layers[layer_dnn], dnn_layers[layer_dnn + 1])
			sequence.append(lin)
			sequence.append(act)
		sequence.append(nn.Linear(dnn_layers[-2], dnn_layers[-1]))

		self.sequence = nn.Sequential(*sequence)
		self.normalized = nn.Softmax(dim=1)

	def forward(self, x):
		features = self.encoder.forward(x)
		raw_probability_values = self.sequence(features)
		probabilities = self.normalized(raw_probability_values)
		# Normalize when using the model.
		return raw_probability_values, probabilities


if __name__ == '__main__':
	classifier = Classifier([384, 50, 10], 32, 2, 16, 128, 5)

	AppLog.info(f'{summary(classifier, input_size=(128, 3, 32, 32))}')
