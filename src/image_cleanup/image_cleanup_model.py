from torch import nn
from torchinfo import summary

from src.common.common_utils import AppLog
from src.encoder_decoder.codec import DecoderLayer, Encoder1stLayer, EncoderLayer

def calc_middle(c_in, c_out, kernel_stack, total_params):
	kernel_sum = sum([sum(ks_in_stack) for ks_in_stack in kernel_stack])
	kernel_count = sum([len(ks_in_stack) for ks_in_stack in kernel_stack])
	kernel_count_layer_1 = len(kernel_stack[0])
	stacks = len(kernel_stack)
	A = 2*kernel_sum + stacks - 1
	B = c_in*(kernel_count_layer_1 +1) + 2*kernel_count + 2*c_out + 1
	C = c_out - total_params
	mid_ch = (-B + (B**2 - 4*A*C)**0.5)/(2*A)
	return mid_ch



class ImageCleanUpEncoder(nn.Module):
	def __init__(self, encoder_channels, stacks_per_layer=None, pool_params=None, kernels_per_layer=None) -> None:
		super().__init__()
		encoder_layers = len(encoder_channels) - 1
		param_estimates = [encoder_channels[x]*encoder_channels[x+1] + encoder_channels[x+1] for x in range(encoder_layers)]
		param_ratios = [param_estimate/param_estimates[0] for param_estimate in param_estimates]
		if kernels_per_layer is None:
			kernels_per_layer = [2 + x for x in range(0, encoder_layers)]
		if stacks_per_layer is None:
			stacks_per_layer = [round(param_ratio*2) for param_ratio in param_ratios]
		if pool_params is None:
			pool_params = 256


		self.encoder = nn.Sequential()
		for channel_index in range(encoder_layers):
			input_channel = encoder_channels[channel_index]
			output_channel = encoder_channels[channel_index + 1]
			stacks = stacks_per_layer[channel_index]
			kernels = kernels_per_layer[channel_index]
			if channel_index == 0:
				encoder_layer = Encoder1stLayer(input_channel, output_channel,pool_params,kernels)
			else:
				middle_channel = round(input_channel/stacks)
				kern_per_stack = [(3,kernels) for _ in range(stacks)]
				encoder_layer = EncoderLayer(input_channel, middle_channel, output_channel, pool_params,kern_per_stack)
			self.encoder.append(encoder_layer)

	def forward(self, x):
		features = self.encoder(x)
		return features


class ImageCleanUpDecoder(nn.Module):
	def __init__(self, decoder_channels) -> None:
		super().__init__()
		decoder_layers = len(decoder_channels) - 1
		self.decoder = nn.Sequential()

		common_kernels = [1,3,5]

		for channel_index in range(decoder_layers):
			input_channel = decoder_channels[channel_index]
			output_channel = decoder_channels[channel_index + 1]
			stacks = (decoder_layers - channel_index)*2 + 2
			mid_ch = input_channel // stacks
			if channel_index < decoder_layers - 1:
				decoder_layer = DecoderLayer(input_channel, mid_ch,output_channel, (stacks,common_kernels))
			else:
				decoder_layer = DecoderLayer(input_channel, input_channel//2, input_channel//2, (1,common_kernels))
			self.decoder.append(decoder_layer)
		inp_ch = decoder_channels[-2] // 2
		out_ch = decoder_channels[-1]
		self.last_layer = nn.Sequential(nn.Conv2d(inp_ch,out_ch, kernel_size=3, padding=1,bias=False),nn.BatchNorm2d(out_ch),nn.Tanh())
		self.decoder_layers = decoder_layers

	def forward(self, x):

		x = self.decoder(x)
		return self.last_layer(x)

class ImageCleanUp(nn.Module):
	def __init__(self, encoder_channels, stacks_per_layer, decoder_channels) -> None:
		super().__init__()
		self.encoder = ImageCleanUpEncoder(encoder_channels, stacks_per_layer)
		self.decoder = ImageCleanUpDecoder(decoder_channels)

	def forward(self, x):
		features = self.encoder(x)
		features = self.decoder(features)
		return features


if __name__ == '__main__':
	# c_in = 64
	# c_out = round(c_in*1.5)
	# kernel_stack = [[3,5],[3,5],[3,5]]
	# total_params = c_in * c_out*9 + c_out
	# mid_val = calc_middle(c_in,c_out,kernel_stack,total_params)
	# mid_ch = round(mid_val)
	# AppLog.info(f"Calculated mid ch : {mid_ch}")
	# ks_n = [(3,2),(3,2),(3,2)]
	en_ch = [3,64,102,161,256]
	en_layer = ImageCleanUpEncoder(en_ch,[2,4,6,8,10])
	dec_ch = [*en_ch]
	dec_ch.reverse()
	cl = ImageCleanUp(en_ch,[2,4,6,8,10],dec_ch)
	# dec_layer = ImageCleanUpDecoder(en_ch)
	summary(cl,input_size=(12,3,256,256))