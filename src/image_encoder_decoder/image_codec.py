import torch
from torch import Tensor, nn
from torch.nn import ModuleDict, functional as F
from torch.nn.functional import interpolate
from torchinfo import summary

from src.common.common_utils import AppLog, Ratio, generate_separated_kernels


def create_sep_kernels(input_channels, output_channels, kernel_size):
	min_channels = min(input_channels, output_channels)
	padding = kernel_size // 2
	conv1 = nn.Conv2d(in_channels=input_channels, out_channels=min_channels, kernel_size=(1, kernel_size),
	                  padding=(0, padding), bias=False)
	conv2 = nn.Conv2d(in_channels=min_channels, out_channels=output_channels, kernel_size=(kernel_size, 1),
	                  padding=(padding, 0), bias=False)
	return conv1, conv2


class EncoderLayer1st(nn.Module):
	def __init__(self, output_channels, kernel_list):
		super().__init__()
		input_channels = 3
		kernels = len(kernel_list)
		kernel_list.sort()

		self.active_path = ModuleDict()
		self.activation = nn.Mish()
		o_ch = output_channels
		kernel_out = round(o_ch * 3 / kernels)

		for i, kernel_size in enumerate(kernel_list):

			padding = kernel_size // 2
			if kernel_size <= 3:
				conv_layer = nn.Conv2d(in_channels=input_channels, out_channels=kernel_out,
				                       kernel_size=kernel_size,
				                       padding=padding, bias=False)

				active_seq = nn.Sequential(conv_layer, nn.BatchNorm2d(kernel_out), self.activation)
				self.active_path[f'{i}'] = active_seq
				continue
			conv1, conv2 = generate_separated_kernels(input_channels, kernel_out, kernel_size,
			                                          Ratio((2 / kernel_size)), switch=False)
			active_seq = nn.Sequential(conv1, conv2, nn.BatchNorm2d(kernel_out), self.activation)

			self.active_path[f'{i}'] = active_seq

			self.h = -1
			self.w = -1

	def set_size(self, h, w):
		self.h = h
		self.w = w

	def forward(self, x):
		convs = []
		for active_path in self.active_path.values():
			active_res = active_path.forward(x)
			convs.append(active_res)
		concat_res = torch.cat(convs, dim=1)
		return F.fractional_max_pool2d(concat_res, kernel_size=2, output_size=[self.h, self.w])


class EncoderBlockWithPassthrough(nn.Module):
	def __init__(self, input_channels, output_channels, kernel_size):
		super().__init__()
		conv1, conv2 = create_sep_kernels(input_channels, output_channels, kernel_size)
		self.active_path = nn.Sequential(conv1, conv2, nn.BatchNorm2d(output_channels), nn.Mish())

	def forward(self, x):
		active_res = self.active_path(x)
		return active_res, x



class EncoderLayer3Conv(nn.Module):
	def __init__(self, input_channels, output_channels, num_kernels):
		super().__init__()
		c1 = input_channels
		c2 = output_channels
		k = num_kernels
		out_p_k = ((c1 ** 0.5) * (c1 * k ** 2 + 36 * c2) ** 0.5 - c1 * k) / (2 * k)

		self.compress = nn.LazyConv2d(out_channels=output_channels, kernel_size=1, padding=0, bias=False)
		self.activation = nn.Mish()
		output_per_kernel = round(out_p_k)

		par_kernels = ModuleDict()
		for i in range(1, num_kernels + 1):
			kernel_size = 2 * i + 1
			activation = EncoderBlockWithPassthrough(input_channels, output_per_kernel, kernel_size)
			par_kernels[f'{i}'] = activation

		self.all_kernels = par_kernels
		self.final_norm = nn.BatchNorm2d(output_channels)

	def forward(self, x, x_compressed):
		evaluated = []
		for act in self.all_kernels.values():
			activated, _ = act(x)
			evaluated.append(activated)
		evaluated.append(x_compressed)
		concat_activated = torch.cat(evaluated, dim=1)
		compress_res = self.compress(concat_activated)
		normed_res = self.final_norm(compress_res)
		active_res = self.activation(normed_res)
		return active_res


class EncoderLayerGrouped(nn.Module):
	def __init__(self, input_channels, output_channels, num_kernels, transition_layer_channels, groups=1):
		super().__init__()
		self.inp_channel_ranges = [round(input_channels * i / groups) for i in range(groups + 1)]
		self.out_channel_ranges = [round(output_channels * i / groups) for i in range(groups + 1)]
		self.compress_input = nn.LazyConv2d(out_channels=input_channels, kernel_size=1, padding=0, bias=False)
		self.compress_input_for_concat_prev = nn.LazyConv2d(out_channels=transition_layer_channels, kernel_size=1,
		                                                    padding=0, bias=False)
		self.encoder_subblocks = ModuleDict()

		for i in range(groups):
			input_channel_per_subblock = self.inp_channel_ranges[i + 1] - self.inp_channel_ranges[i]
			output_channel_per_subblock = self.out_channel_ranges[i + 1] - self.out_channel_ranges[i]
			self.encoder_subblocks[f'group{i}'] = EncoderLayer3Conv(input_channel_per_subblock,
			                                                        output_channel_per_subblock, num_kernels)
			self.h = -1
			self.w = -1

	def set_size(self, h, w):
		self.h = h
		self.w = w

	def forward(self, x, prev_layers):
		x_compressed = self.compress_input_for_concat_prev(x)
		x_compress_for_activation = self.compress_input(torch.cat([x, prev_layers], dim=1))
		evaluated = []
		for i, subblock in enumerate(self.encoder_subblocks.values()):
			encoder_input = x_compress_for_activation[:, self.inp_channel_ranges[i]:self.inp_channel_ranges[i + 1], :,
			                :]
			activated = subblock(encoder_input, x_compressed)
			evaluated.append(activated)
		activated = torch.cat(evaluated, dim=1)
		inactive = torch.cat([x_compressed, prev_layers], dim=1)
		down_sample_active = F.fractional_max_pool2d(activated, kernel_size=2, output_size=[self.h, self.w])
		down_sample_inactive = F.interpolate(inactive, size=[self.h, self.w], mode='bilinear')

		return down_sample_active, down_sample_inactive


class Encoder(nn.Module):
	def __init__(self, ch_in, ch_out, layers, total_downsample=1 / 16):
		super().__init__()
		ratio = (ch_out / ch_in) ** (1 / (layers - 1))
		channels = [round(ch_in * ratio ** i) for i in range(layers)]
		layer1_compute = channels[0] * channels[1]
		downsample_ratio = total_downsample ** (1 / layers)
		self.layers = ModuleDict()
		self.transition_layers = ModuleDict()
		self.downsample_input = nn.UpsamplingBilinear2d(scale_factor=downsample_ratio)
		for i in range(layers):
			if i == 0:
				self.layers[f'{i}'] = EncoderLayer1st(channels[i], [1, 3, 5, 7, 9])
				continue
			compute_cost = channels[i - 1] * channels[i] / layer1_compute
			groups = round(compute_cost ** 0.5)
			AppLog.info(f'Proportional compute load {compute_cost:.1f}')
			self.layers[f'{i}'] = EncoderLayerGrouped(channels[i - 1], channels[i], 3, 10,
			                                          groups=groups)
		self.layer_count = layers
		self.activation = nn.Mish()
		self.ch_out = ch_out
		self.downsample_ratio = downsample_ratio

	def forward(self, x):
		x = (x - 0.5) * 2
		_, _, h, w = x.shape
		ds = self.downsample_ratio
		sizes_down = [(round(h * (ds ** layer)), round(w * (ds ** layer))) for layer in
		              range(1, self.layer_count + 1)]
		inputs = torch.empty((x.shape[0], 0, x.shape[2], x.shape[3])).to(x.device)
		for i, layer in enumerate(self.layers.values()):
			layer.set_size(sizes_down[i][0], sizes_down[i][1])
			if i == 0:
				x = layer(x)
				inputs = torch.empty((x.shape[0], 0, x.shape[2], x.shape[3])).to(x.device)
				continue
			x, inputs = layer(x, inputs)

		# x = self.activation(x)
		reduced = x / (torch.abs(x) + 1)
		return reduced


class ImageDecoderLayer(nn.Module):
	def __init__(self, input_channels, output_channels, cardinality, upscale=2):
		super().__init__()
		inp_ch = round(input_channels / cardinality)
		out_ch = round(output_channels / cardinality)
		self.conv_layers = ModuleDict()
		self.passthrough = nn.LazyConv2d(output_channels, kernel_size=1, padding=0, bias=False)
		for i in range(1, cardinality + 1):
			conv_lower = nn.Conv2d(in_channels=input_channels, out_channels=inp_ch, kernel_size=1, padding=0,
			                       bias=False)
			conv_layer = nn.Conv2d(in_channels=inp_ch, out_channels=out_ch, kernel_size=3, padding=1, bias=False)
			seq = nn.Sequential(conv_lower, conv_layer)
			self.conv_layers[f'{i}'] = seq
		self.conv = nn.LazyConv2d(output_channels, kernel_size=1, padding=0, bias=False)
		self.norm = nn.InstanceNorm2d(output_channels)
		self.activation = nn.Mish()
		# self.upscale = nn.Upsample(scale_factor=upscale, mode='bicubic')
		self.h = 0
		self.w = 0

	def set_size(self, h, w):
		self.h = h
		self.w = w

	def forward(self, x):
		upscaled = interpolate(x, size=(self.h, self.w), mode='bicubic')
		convs = []
		for conv_layer in self.conv_layers.values():
			conv: Tensor = conv_layer.forward(upscaled)
			convs.append(conv)
		all_convs = torch.cat(convs, dim=1)
		passthrough = self.passthrough(upscaled)
		conv_res = self.conv(all_convs)
		normed_res = self.norm(conv_res)
		active_res = self.activation(normed_res)
		return active_res + passthrough


class Decoder(nn.Module):
	def __init__(self, ch_in, ch_out, layers, total_upsample=16.0) -> None:
		super().__init__()
		ratio = (ch_out / ch_in) ** (1 / (layers - 1))
		channels = [round(ch_in * ratio ** i) for i in range(layers)]
		channels = [*channels, 3]
		layers = len(channels) - 1
		self.layers = layers
		upsample_ratio = total_upsample ** (1 / layers)
		decoder_layers = ModuleDict()
		for layer in range(layers):
			ch_in = channels[layer]
			ch_out = channels[layer + 1]
			dec_layer = ImageDecoderLayer(ch_in, ch_out, cardinality=max(1, (layers - layer) // 3),
			                              upscale=upsample_ratio)
			decoder_layers[f'{layer}'] = dec_layer

		self.decoder_layers = decoder_layers
		self.size = [128, 128]
		self.last_activation = nn.Sigmoid()
		self.upsample_ratio = upsample_ratio
		AppLog.info(f'Decoder upsample : {self.upsample_ratio}')

	def set_size(self, h, w):
		self.size = [h, w]

	def forward(self, latent_z):
		[h, w] = self.size
		_, _, zh, zw = latent_z.shape
		upsample_h = (h / zh) ** (1 / self.layers)
		upsample_w = (w / zw) ** (1 / self.layers)
		sizes_up = [(round(zh * (upsample_h ** layer)), round(zw * (upsample_w ** layer))) for layer in
		            range(1, self.layers + 1)]

		z_m = latent_z

		for i, dec_layer in enumerate(self.decoder_layers.values()):
			dec_layer.set_size(sizes_up[i][0], sizes_up[i][1])
			z_m = dec_layer.forward(z_m)

		# x = interpolate(z_m, size=(h, w), mode='bilinear', align_corners=False)

		return self.last_activation(z_m)


class ImageCodec(nn.Module):
	def __init__(self, enc_chin, latent_channels, dec_chout, enc_layers=4, dec_layers=4, downsample=1 / 16):
		super().__init__()
		self.encoder = Encoder(enc_chin, latent_channels, layers=enc_layers, total_downsample=downsample)
		self.decoder = Decoder(latent_channels, dec_chout, layers=dec_layers, total_upsample=(1 / downsample))

	def forward(self, x):
		latent = self.encoder.forward(x)
		self.decoder.set_size(x.shape[2], x.shape[3])
		final_res = self.decoder.forward(latent)
		return final_res


if __name__ == '__main__':
	# chn =[64, 128, 192, 256]
	# chn.reverse()
	# dec = Encoder(chn)
	# enc = Encoder(64, 256, 6, 1 / 16)
	enc = ImageCodec(64, 256, 64, 8, 6)
	# AppLog.info(f'Encoder : {enc}')
	summary(enc, [(12, 3, 256, 288)])
