from torch import nn

from src.encoder_decoder.codec import DecoderLayer, EncoderLayer


class ImageCleanUpEncoder(nn.Module):
	def __init__(self, encoder_channels, total_downsample) -> None:
		super().__init__()
		encoder_layers = len(encoder_channels) - 1
		downsample_ratio = total_downsample ** (-1 / encoder_layers)

		self.encoder = nn.Sequential()
		for channel_index in range(encoder_layers):
			input_channel = encoder_channels[channel_index]
			output_channel = encoder_channels[channel_index + 1]
			if channel_index == 0:
				encoder_layer = EncoderLayer(input_channel, output_channel, [(3, 1)], downsample=downsample_ratio)
			else:
				encoder_layer = EncoderLayer(input_channel, output_channel, [(3, 1), (5, 3 / 5), (7, 3 / 7)],
											 downsample=downsample_ratio)
			self.encoder.append(encoder_layer)

	def forward(self, x):
		features = self.encoder(x)
		return features


class ImageCleanUpDecoder(nn.Module):
	def __init__(self, decoder_channels, upsample) -> None:
		super().__init__()
		decoder_layers = len(decoder_channels) - 1
		self.decoder = nn.ModuleDict()

		self.decoder_activation = nn.Mish()
		self.decoder_activation_last = nn.Softplus()
		upsample_ratio = upsample ** (-1 / decoder_layers)

		for channel_index in range(decoder_layers):
			input_channel = decoder_channels[channel_index]
			output_channel = decoder_channels[channel_index + 1]
			if channel_index == decoder_layers - 1:
				decoder_layer = DecoderLayer(input_channel, output_channel, [(3, 1)], upsample_ratio)
			else:
				decoder_layer = DecoderLayer(input_channel, output_channel, [(3, 1), (5, 1)], upsample_ratio)
			self.decoder[f'd_{channel_index}'] = decoder_layer
		self.decoder_layers = decoder_layers

	def forward(self, x):

		for i, layer in enumerate(self.decoder.values()):
			x = layer(x)
			if i != self.decoder_layers - 1:
				# Use resnet style skip connection
				x = self.decoder_activation(x)+x
			else:
				x = self.decoder_activation_last(x)
		return x


class ImageCleanupModel(nn.Module):
	def __init__(self, encoder_channels, decoder_channels, total_downsample):
		super().__init__()

		self.middle_depthwise_conv = nn.Conv2d(encoder_channels[-1], decoder_channels[0], kernel_size=1, padding=(1,
																												  1))
