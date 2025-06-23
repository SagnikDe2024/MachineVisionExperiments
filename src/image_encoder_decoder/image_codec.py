import torch
from torch import Tensor, nn
from torch.nn import ModuleDict
from torch.nn.functional import interpolate
from torchinfo import summary


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
		total_weights = kernels * (kernels - 1) / 2
		self.inactive_path = ModuleDict()
		self.active_path = ModuleDict()
		self.activation = nn.Mish()
		o_ch = output_channels
		self.final_norm = nn.BatchNorm2d(o_ch)
		self.final_conv = nn.LazyConv2d(out_channels=o_ch, kernel_size=1, padding=0, bias=False)
		for i, kernel_size in enumerate(kernel_list):
			max_expand_channel = o_ch * 1
			output_channel_compress = max_expand_channel * (kernels - i) / total_weights
			comp_active = round(output_channel_compress * 5 / 8)
			comp_inactive = round(output_channel_compress * 3 / 8)
			compress_active = nn.Conv2d(in_channels=o_ch, out_channels=comp_active,
			                            kernel_size=1, bias=False)
			compress_inactive = nn.Conv2d(in_channels=o_ch, out_channels=comp_inactive,
			                              kernel_size=1, bias=False)

			padding = kernel_size // 2
			if kernel_size <= 3:
				conv_layer = nn.Conv2d(in_channels=input_channels, out_channels=o_ch,
				                       kernel_size=kernel_size,
				                       padding=padding, bias=False)

				active_seq = nn.Sequential(conv_layer, nn.BatchNorm2d(o_ch), self.activation,
				                           compress_active)
				self.active_path[f'{i}'] = active_seq
				inactive_seq = nn.Sequential(conv_layer, compress_inactive)
				self.inactive_path[f'{i}'] = inactive_seq
				continue
			conv1, conv2 = create_sep_kernels(input_channels, o_ch, kernel_size)
			seq = nn.Sequential(conv1, conv2)
			active_seq = nn.Sequential(seq, nn.BatchNorm2d(o_ch), self.activation, compress_active)
			inactive_seq = nn.Sequential(seq, compress_inactive)

			self.active_path[f'{i}'] = active_seq
			self.inactive_path[f'{i}'] = inactive_seq

	def forward(self, x):
		convs = []
		for inactive_path, active_path in zip(self.inactive_path.values(), self.active_path.values()):
			conv: Tensor = inactive_path.forward(x)
			convs.append(conv)
			active_res = active_path.forward(x)
			convs.append(active_res)
		concat_res = torch.cat(convs, dim=1)
		compress_res = self.final_conv(concat_res)
		normed_res = self.final_norm(compress_res)
		active_res = self.activation(normed_res)
		return active_res


class EncoderLayer3Conv(nn.Module):
	def __init__(self, input_channels, output_channels, kernels, cardinality):
		super().__init__()
		self.compress = nn.LazyConv2d(out_channels=output_channels, kernel_size=1, padding=0, bias=False)
		self.activation = nn.Mish()
		total_groups = cardinality * kernels
		input_per_kernel = round(input_channels / total_groups)
		output_per_kernel = round(output_channels / total_groups)
		par_kernels = ModuleDict()
		for i in range(1, kernels + 1):
			for j in range(cardinality):
				conv_lower = nn.Conv2d(in_channels=input_channels, out_channels=input_per_kernel, kernel_size=1,
				                       padding=0, bias=False)
				conv_layer = nn.Conv2d(in_channels=input_per_kernel, out_channels=output_per_kernel, kernel_size=3,
				                       padding=i, bias=False, dilation=i)
				all_ops = nn.Sequential(conv_lower, conv_layer, nn.BatchNorm2d(output_per_kernel), nn.Mish())
				par_kernels[f'{i}_{j}'] = all_ops
		self.all_kernels = par_kernels
		self.final_norm = nn.BatchNorm2d(output_channels)

	def forward(self, x):
		evaluated = []
		for kernel_name, kernel in self.all_kernels.items():
			r = kernel(x)
			evaluated.append(r)
		evaluated.append(x)
		concat_res = torch.cat(evaluated, dim=1)
		compress_res = self.compress(concat_res)
		normed_res = self.final_norm(compress_res)
		active_res = self.activation(normed_res)
		return active_res


class Encoder(nn.Module):
	def __init__(self, ch_in, ch_out, layers, total_downsample=1 / 16):
		super().__init__()
		ratio = (ch_out / ch_in) ** (1 / (layers - 1))
		channels = [round(ch_in * ratio ** i) for i in range(layers)]
		downsample_ratio = total_downsample ** (1 / (layers - 1))
		self.layers = ModuleDict()
		for i in range(layers):
			if i == 0:
				self.layers[f'{i}'] = EncoderLayer1st(channels[i], [3, 5, 7])
				continue
			self.layers[f'{i}'] = EncoderLayer3Conv(channels[i - 1], channels[i], 3, 1)
		self.reduce = nn.FractionalMaxPool2d(2, output_ratio=downsample_ratio)
		self.activation = nn.Mish()

	def forward(self, x):

		x_orig_max = torch.amax(torch.abs(x), dim=(1, 2, 3), keepdim=True)
		for layer in self.layers.values():
			x = layer(x)
			x = self.reduce(x)
		x_abs = torch.abs(x)
		x_max = torch.amax(x_abs, dim=(1, 2, 3), keepdim=True)
		return x / x_max, x_orig_max


class ImageDecoderLayer(nn.Module):
	def __init__(self, input_channels, output_channels, cardinality, upscale=2):
		super().__init__()
		inp_ch = round(input_channels / cardinality)
		out_ch = round(output_channels / cardinality)
		self.conv_layers = ModuleDict()
		for i in range(1, cardinality + 1):
			conv_lower = nn.Conv2d(in_channels=input_channels, out_channels=inp_ch, kernel_size=1, padding=0,
			                       bias=False)
			conv_layer = nn.Conv2d(in_channels=inp_ch, out_channels=out_ch, kernel_size=3, padding=1, bias=False)
			seq = nn.Sequential(conv_lower, conv_layer)
			self.conv_layers[f'{i}'] = seq
		self.conv = nn.LazyConv2d(output_channels, kernel_size=1, padding=0, bias=False)
		self.norm = nn.InstanceNorm2d(output_channels)
		self.activation = nn.Mish()
		self.upscale = nn.Upsample(scale_factor=upscale, mode='bicubic')

	def forward(self, x):
		upscaled = self.upscale(x)
		convs = []
		for conv_layer in self.conv_layers.values():
			conv: Tensor = conv_layer.forward(upscaled)
			convs.append(conv)
		all_convs = torch.cat(convs, dim=1)
		conv_res = self.conv(all_convs)
		normed_res = self.norm(conv_res)
		active_res = self.activation(normed_res)
		return active_res, normed_res


class Decoder(nn.Module):
	def __init__(self, ch_in, ch_out, layers, total_upsample=16.0) -> None:
		super().__init__()
		ratio = (ch_out / ch_in) ** (1 / (layers - 1))
		channels = [round(ch_in * ratio ** i) for i in range(layers)]
		channels = [*channels, 3]
		layers = len(channels) - 1
		self.layers = layers
		upsample_ratio = total_upsample ** (1 / (layers - 1))
		decoder_layers = ModuleDict()
		for layer in range(layers):
			ch_in = channels[layer]
			ch_out = channels[layer + 1]
			dec_layer = ImageDecoderLayer(ch_in, ch_out, cardinality=(layers - layer), upscale=upsample_ratio)
			decoder_layers[f'{layer}'] = dec_layer

		self.decoder_layers = decoder_layers
		self.size = [128, 128]

	def set_size(self, h, w):
		self.size = [h, w]

	def forward(self, latent_z, maxes):
		[h, w] = self.size

		z_m = latent_z
		for dec_layer in self.decoder_layers.values():
			z_m, unactivated_z_m = dec_layer.forward(z_m)

		x = interpolate(z_m, size=(h, w), mode='bilinear', align_corners=False)
		x_abs = torch.abs(x)
		x_max = torch.amax(x_abs, dim=(1, 2, 3), keepdim=True)
		x_normed = x / x_max
		x_max_new = torch.mul(x_normed, maxes)

		return x_max_new


class ImageCodec(nn.Module):
	def __init__(self, enc_chin, latent_channels, dec_chout, enc_layers=4, dec_layers=4, downsample=1 / 16):
		super().__init__()
		self.encoder = Encoder(enc_chin, latent_channels, layers=enc_layers, total_downsample=downsample)
		self.decoder = Decoder(latent_channels, dec_chout, layers=dec_layers, total_upsample=(1 / downsample))

	def forward(self, x):
		latent, maxes = self.encoder.forward(x)
		self.decoder.set_size(x.shape[2], x.shape[3])
		final_res = self.decoder.forward(latent, maxes)
		return final_res


if __name__ == '__main__':
	# chn =[64, 128, 192, 256]
	# chn.reverse()
	# dec = Encoder(chn)
	enc = ImageCodec(64, 256, 64, 6, 5)
	summary(enc, [(11, 3, 288, 288)])
