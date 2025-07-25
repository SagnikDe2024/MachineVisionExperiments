import torch
from torch import Tensor, nn
from torch.nn import ModuleDict, functional as F
from torch.nn.functional import interpolate
from torchinfo import summary

from src.common.common_utils import AppLog
from src.encoder_decoder.codec import CodecMultiKernelStack, create_sep_kernels


class EncoderLayer1st(nn.Module):
	def __init__(self, output_channels, kernel_list):
		super().__init__()
		input_channels = 3
		kernels = len(kernel_list)
		kernel_list.sort()

		self.active_path = ModuleDict()
		self.activation = nn.Mish()
		# o_ch = output_channels*3/4
		# kernel_out = round(o_ch * 3 / kernels)
		# AppLog.info(f'Kernel out {kernel_out}')
		AppLog.info(f'Output channel {output_channels}')

		for i, kernel_size in enumerate(kernel_list):
			padding = kernel_size // 2
			if kernel_size == 1:
				out_ch = round(output_channels / 4)
			elif kernel_size == 3:
				out_ch = round(output_channels * 3 / 4)
			else:
				out_ch = round(((3 / kernel_size) ** 2) * output_channels)
			conv_layer = nn.Conv2d(in_channels=input_channels, out_channels=out_ch, kernel_size=kernel_size,
			                       padding=padding, bias=False)
			active_seq = nn.Sequential(conv_layer, nn.BatchNorm2d(out_ch), self.activation)
			self.active_path[f'{i}'] = active_seq

		self.h = -1
		self.w = -1
		self.down_sample_ratio = -1

	def set_size(self, h, w):
		self.h = h
		self.w = w

	def set_down_sample_ratio(self, ratio):
		self.down_sample_ratio = ratio

	def forward(self, x):
		convs = []
		for active_path in self.active_path.values():
			active_res = active_path.forward(x)
			convs.append(active_res)
		concat_res = torch.cat(convs, dim=1)
		if 0 < self.down_sample_ratio < 1:
			return F.fractional_max_pool2d(concat_res, kernel_size=2,
			                               output_ratio=(self.down_sample_ratio, self.down_sample_ratio))
		else:
			return F.fractional_max_pool2d(concat_res, kernel_size=2, output_size=[self.h, self.w])


class EncoderBlockWithPassthrough(nn.Module):
	def __init__(self, input_channels, output_channels, kernel_size):
		super().__init__()
		conv1, conv2 = create_sep_kernels(input_channels, output_channels, kernel_size)
		self.active_path = nn.Sequential(conv1, conv2, nn.BatchNorm2d(output_channels), nn.Mish())

	def forward(self, x):
		active_res = self.active_path(x)
		return active_res, x


# Let c_out/c_in = j where j > 1 (for an encoder)
# Now for compute reasons the features are compressed to m features, convolved, normed, activated and the output is squeezed again.
# Let c <- c_in and c j <- c_out.
# The operation is as follows:
# c [1 x 1] m -> m [ks x 1] m -> m [1 x ks] (m) -> batchnorm -> activate -> (m) [1 x 1 + 1 (for bias)] (c j/n)
# Here ks is the kernel size, ks = 2*k-1, where k = 0,1,2 ... for ks as 1,3,5 etc.
# Total learnable params = c*m + m^2*ks + m j c/n  + batchnorm (2 * m) +  (j c / n)

# If there are n kernels we have
# c*m*n + 2*m^2*(ks_1 + ks_2 ... ks_n) + 2*m*n (batchnorm) + m*j*c + j*c (bias term)
# => 2*m^2*(ks_1 + ks_2 ... ks_n) + m*(c*n + 2*n + j*c) + j*c
# Let P we the total number of params we need. Then we have
# 2*m^2*(ks_1 + ks_2 ... ks_n) + m*(c*n + 2*n + j*c) + j*c == P
# This is a of type quadratic eq. A m^2 + B m + C == 0
# Here A = 2*(ks_1 + ks_2 ... ks_n), B = (c*n + 2*n + j*c) , C = j*c - P


def calculate_intermediate_ch(input_channels, kernels, max_params, output_channels):
	A = 2*(sum(kernels))
	n = len(kernels)
	c = input_channels
	j = output_channels / input_channels
	B = (c * n + 2 * n + j * c)
	C = j * c - max_params
	m = (-B + (B ** 2 - 4 * A * C) ** 0.5) / (2 * A)
	m_in = round(m)
	return m_in, n


def getEncoderNKernelsBlock(input_channels, output_channels, kernels : list[int], max_params=-1):
	if max_params == -1:
		max_params = 9 * input_channels * output_channels
	max_params = max_params - (
				input_channels * output_channels + output_channels) if input_channels != output_channels else max_params
	m_in, n = calculate_intermediate_ch(input_channels, kernels, max_params, output_channels)


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
		self.out_channel_ranges = [round(output_channels * i / groups) for i in range(groups + 1)]
		self.compress_input_for_concat_prev = nn.LazyConv2d(out_channels=transition_layer_channels, kernel_size=1,
		                                                    padding=0, bias=False)
		self.encoder_subblocks = ModuleDict()
		self.encoder_input_conv = ModuleDict()
		input_channel_per_subblock = input_channels // groups
		for i in range(groups):
			output_channel_per_subblock = self.out_channel_ranges[i + 1] - self.out_channel_ranges[i]
			self.encoder_input_conv[f'subblock_inp{i}'] = nn.LazyConv2d(out_channels=input_channel_per_subblock,
			                                                            kernel_size=1, padding=0, bias=False)
			self.encoder_subblocks[f'group{i}'] = EncoderLayer3Conv(input_channel_per_subblock,
			                                                        output_channel_per_subblock, num_kernels)
		self.h = -1
		self.w = -1
		self.down_sample_ratio = -1

	def set_size(self, h, w):
		self.h = h
		self.w = w

	def set_down_sample_ratio(self, ratio):
		self.down_sample_ratio = ratio

	def forward(self, x, prev_layers):
		x_compressed = self.compress_input_for_concat_prev(x)
		# x_for_activation = torch.cat([x, prev_layers], dim=1)
		evaluated = []
		for i, (inp_conv, subblock) in enumerate(
				zip(self.encoder_input_conv.values(), self.encoder_subblocks.values())):
			encoder_input = inp_conv(torch.cat([x, prev_layers], dim=1))
			activated = subblock(encoder_input, x_compressed)
			evaluated.append(activated)
		activated = torch.cat(evaluated, dim=1)
		inactive = torch.cat([x_compressed, prev_layers], dim=1)
		if 0 < self.down_sample_ratio < 1:
			down_sample_active = F.fractional_max_pool2d(activated, kernel_size=2,
			                                             output_ratio=(self.down_sample_ratio, self.down_sample_ratio))
			down_sample_inactive = F.interpolate(inactive, scale_factor=self.down_sample_ratio, mode='bilinear')

		else:
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
				self.layers[f'{i}'] = EncoderLayer1st(channels[i], [1, 3, 5, 7])
				continue
			compute_cost = channels[i - 1] * channels[i] / layer1_compute
			groups = 2 if i == 1 else round(compute_cost ** 0.5)
			AppLog.info(f'Proportional compute load {compute_cost:.1f}')
			self.layers[f'{i}'] = EncoderLayerGrouped(channels[i - 1], channels[i], 3, 9, groups=groups)
		self.layer_count = layers
		self.activation = nn.Mish()
		self.ch_out = ch_out
		self.downsample_ratio = downsample_ratio

	def forward(self, x):
		_, _, h, w = x.shape
		ds = self.downsample_ratio
		sizes_down = [(round(h * (ds ** layer)), round(w * (ds ** layer))) for layer in range(1, self.layer_count + 1)]
		inputs = torch.empty((x.shape[0], 0, x.shape[2], x.shape[3])).to(x.device)
		for i, layer in enumerate(self.layers.values()):
			layer.set_size(sizes_down[i][0], sizes_down[i][1])
			# layer.set_down_sample_ratio(ds)
			if i == 0:
				x = layer(x)
				inputs = torch.empty((x.shape[0], 0, x.shape[2], x.shape[3])).to(x.device)
				continue
			x, inputs = layer(x, inputs)

		# x = self.activation(x)
		return x


class ImageDecoderLayer(nn.Module):
	def __init__(self, input_channels, output_channels, transition, cardinality):
		super().__init__()
		inp_ch = round(input_channels / cardinality)
		out_ch = round(output_channels / cardinality)
		self.transition_conv = nn.Conv2d(in_channels=input_channels, out_channels=transition, kernel_size=1, padding=0,
		                                 bias=False)
		self.conv_layers = ModuleDict()
		for i in range(1, cardinality + 1):
			conv_lower = nn.LazyConv2d(out_channels=inp_ch, kernel_size=1, padding=0, bias=False)
			conv_layer = nn.Conv2d(in_channels=inp_ch, out_channels=out_ch, kernel_size=3, padding=1, bias=False)
			seq = nn.Sequential(conv_lower, conv_layer)
			self.conv_layers[f'{i}'] = seq
		self.conv = nn.LazyConv2d(output_channels, kernel_size=1, padding=0, bias=False)
		self.norm = nn.BatchNorm2d(output_channels)
		self.activation = nn.Mish()
		self.upsample_ratio = -1
		self.h = 0
		self.w = 0

	def set_size(self, h, w):
		self.h = h
		self.w = w

	def set_upsample_ratio(self, ratio):
		self.upsample_ratio = ratio

	def upscale_tensor(self, tensor):
		upscaled = interpolate(tensor, scale_factor=self.upsample_ratio,
		                       mode='bicubic') if self.upsample_ratio > 1 else interpolate(tensor,
		                                                                                   size=(self.h, self.w),
		                                                                                   mode='bicubic')
		return upscaled

	def forward(self, x, prev):

		transition_x_inp = self.transition_conv(x)
		upscaled = self.upscale_tensor(torch.cat([x, prev], dim=1))
		transition_x = self.upscale_tensor(torch.cat([transition_x_inp, prev], dim=1))

		convs = []
		for conv_layer in self.conv_layers.values():
			conv: Tensor = conv_layer.forward(upscaled)
			convs.append(conv)
		all_convs = torch.cat(convs, dim=1)
		conv_res = self.conv(torch.cat([all_convs, upscaled], dim=1))
		normed_res = self.norm(conv_res)
		active_res = self.activation(normed_res)

		return active_res, transition_x


class Decoder(nn.Module):
	def __init__(self, ch_in, ch_out, layers, total_upsample=16.0) -> None:
		super().__init__()
		ratio = (ch_out / ch_in) ** (1 / (layers - 1))
		channels = [round(ch_in * ratio ** i) for i in range(layers)]
		AppLog.info(f'Decoder channels {channels}')
		param_compute = channels[-2] * channels[-1]*(1/3)
		channels = [*channels, 3]
		layers = len(channels) - 1
		self.layers = layers
		upsample_ratio = total_upsample ** (1 / layers)

		decoder_layers = ModuleDict()
		for layer in range(layers):
			ch_in = channels[layer]
			ch_out = channels[layer + 1]
			cardinality = max(round((ch_in * ch_out / param_compute) ** 0.5), 1)
			dec_layer = ImageDecoderLayer(ch_in, ch_out, 9, cardinality=cardinality)
			decoder_layers[f'{layer}'] = dec_layer

		self.decoder_layers = decoder_layers
		self.size = [128, 128]
		self.last_activation = nn.Tanh()
		self.upsample_ratio = upsample_ratio
		self.last_compress = nn.LazyConv2d(out_channels=3, kernel_size=1, padding=0)
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
		prev_results = torch.empty((z_m.shape[0], 0, z_m.shape[2], z_m.shape[3])).to(z_m.device)

		for i, dec_layer in enumerate(self.decoder_layers.values()):
			dec_layer.set_size(sizes_up[i][0], sizes_up[i][1])
			# dec_layer.set_upsample_ratio(self.upsample_ratio)
			z_m, new_prev = dec_layer(z_m, prev_results)
			prev_results = new_prev

		compressed_prev = self.last_compress(prev_results)
		return self.last_activation(z_m + compressed_prev)


class ImageCodec(nn.Module):
	def __init__(self, enc_chin, latent_channels, dec_chout, enc_layers=4, dec_layers=4, downsample=1 / 16):
		super().__init__()
		self.encoder = Encoder(enc_chin, latent_channels, layers=enc_layers, total_downsample=downsample)
		self.decoder = Decoder(latent_channels, dec_chout, layers=dec_layers, total_upsample=(1 / downsample))

	def forward(self, x):
		latent = self.encoder.forward(x)
		self.decoder.set_size(x.shape[2], x.shape[3])
		final_res = self.decoder(latent)
		return final_res, latent


def prepare_encoder_data(data):
	# mean = torch.mean(data, dim=(2, 3), keepdim=True)
	# abs_diff = torch.abs(data - mean)
	# abs_diff_mean = torch.mean(abs_diff, dim=(2, 3), keepdim=True)
	# return (data - mean) / (abs_diff_mean + 1e-7)
	return (data - 0.5) / 0.5


def scale_decoder_data(data):
	sc1 = data * 256 / 255
	sc2 = sc1*0.5+0.5
	return sc2


def encode_decode_from_model(model, data):
	data = prepare_encoder_data(data)
	final_res, latent = model(data)
	final_res = scale_decoder_data(final_res)
	return final_res, latent

def calcParamsForMid(in_ch,out_ch, kernek_stack : list[list[int]], param_max):
	k1 = len(kernek_stack[0])
	kc = sum( len(ks) for ks in kernek_stack )
	ks = sum( sum(ks) for ks in kernek_stack )
	l = len(kernek_stack)

	B = (in_ch*k1 + in_ch + 2*kc + 2*out_ch + 1)
	A = (2*ks + l - 1)
	C = out_ch - param_max
	m = (-B + (B**2 - 4*A*C) ** 0.5) / (2*A)
	return m


if __name__ == '__main__':

	in_ch = 161
	out_ch = 256

	m = calcParamsForMid(in_ch, out_ch, [[1, 3, 5] for _ in range(3)], 25 * in_ch * out_ch)
	mid_ch = round(m)
	AppLog.info(f'Encoder middle channel : {m} , {mid_ch}')


	stack_layer = CodecMultiKernelStack(in_ch, mid_ch, out_ch, (3, [1, 3, 5]))
	# stack_layer = EncoderNKernelsBlock(161, 256, [1, 3, 5], 20)
	AppLog.info(f'Stacked Encoder : {stack_layer}')
	summary(stack_layer, [(12, in_ch, 40, 40)])
	# poolS = PoolSelector(127)
	# summary(poolS, [(12, 256, 40, 40)])

	# For stacked.
	# m (c k_1 + c + 2 k_c + 2 o + 1) + m^2 (2 k_s + l - 1) + o

	# chn =[64, 128, 192, 256]
	# chn.reverse()
	# dec = Encoder(chn)
	# enc = Encoder(64, 256, 6, 1 / 16)
	# enc = ImageCodec(64, 256, 64, 7, 6,downsample=(256*2/3)**(-0.5))
	# AppLog.info(f'Encoder : {enc}')
	# summary(enc, [(12, 3, 256, 256)])
