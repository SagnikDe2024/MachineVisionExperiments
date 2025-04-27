from torch import nn

from src.encoder_decoder.codec import DecoderLayer, EncoderLayer


class ImageCleanupModel(nn.Module):
	def __init__(self, encoder_channels, decoder_channels, total_downsample):
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

		self.middle_depthwise_conv = nn.Conv2d(encoder_channels[-1], decoder_channels[0], kernel_size=1, padding=(1,
																												  1))

		self.decoder = nn.ModuleDict()

		decoder_layers = len(decoder_channels) - 1
		decoder_activation = nn.Mish()
		decoder_activation_last = nn.Softplus()

		for channel_index in range(decoder_layers):
			input_channel = decoder_channels[channel_index]
			output_channel = decoder_channels[channel_index + 1]
			if channel_index == decoder_layers - 1:
				decoder_layer = DecoderLayer(input_channel, output_channel, [(3, 1)])
			else:
				decoder_layer = DecoderLayer(input_channel, output_channel, [(3, 1), (5, 1)])

			self.decoder[f'd_{channel_index}'] = decoder_layer


