from math import lcm, ceil

import torch
from torch import nn
from torch.nn import ModuleDict
from torch.nn.functional import interpolate
from torchinfo import summary

from src.common.common_utils import AppLog


class SimpleDenseLayer(nn.Module):

	def __init__(self, in_ch,mid_ch, out_ch, depth ,groups ,normed=True,in_groups=1,out_groups=1):
		super().__init__()
		divisibility = lcm(4, groups)
		mid_ch = round(mid_ch / divisibility) * divisibility
		in_div = lcm(4, groups, in_groups)
		in_ch = round(in_ch / in_div) * in_div
		out_div = lcm(4, groups, out_groups)
		out_ch = round(out_ch / out_div) * out_div

		self.mid_ch = mid_ch
		self.out_ch = out_ch
		self.in_ch = in_ch
		self.inp_conv = nn.LazyConv2d(out_channels=in_ch, kernel_size=1, padding=0, bias=True,groups=in_groups)
		self.mid_conv_modules = ModuleDict()
		use_bias = not normed
		cost = 0
		kernel_list= [3 for _ in range(depth)]
		for i, k in enumerate(kernel_list):
			conv = nn.LazyConv2d(out_channels=mid_ch, kernel_size=k, padding=k // 2, bias=use_bias, groups=groups,
			                     padding_mode='reflect')
			cost += (mid_ch * k / groups) ** 2 * (i + 1) + mid_ch * 2
			norm = nn.GroupNorm(groups, mid_ch) if normed else nn.Identity()
			shuffle = nn.ChannelShuffle(2)
			act = nn.Mish()
			self.mid_conv_modules[f'{k}'] = nn.Sequential(shuffle, conv, shuffle, norm, act)
		self.out_conv = nn.LazyConv2d(out_channels=out_ch, kernel_size=1, padding=0, bias=True,groups=out_groups)
		cost += out_ch * 2 + mid_ch * len(kernel_list)
		AppLog.info(f'Dense layer in_ch={self.in_ch} mid_ch={self.mid_ch}, out_ch={self.out_ch}, groups={groups}, kernels={kernel_list}')
		AppLog.info(f'Approx params {cost}')
		AppLog.info('------------------------------------')

	def forward(self, x):
		dense_input = self.inp_conv(x)

		for mid_conv in self.mid_conv_modules.values():
			mid_conv_res = mid_conv(dense_input)
			dense_input = torch.cat([mid_conv_res,dense_input], dim=1)
		out = self.out_conv(dense_input)
		return out


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






class Encoder(nn.Module):
	def __init__(self, ch_in_enc, ch_out_enc, layers=5):
		super().__init__()
		channels, depths, mid_chs, layer_group = calc_channels_depth_and_midchs(ch_in_enc, ch_out_enc, 8, 16, layers)
		AppLog.info(f'Encoder channels {channels}, depths {depths}, mid_chs {mid_chs}')

		self.layers = ModuleDict()
		self.downsample_input = nn.PixelUnshuffle(2)
		for l_i in range(layers):
			# total_calc = ch_in * mid_ch_calc + (s * (s + 1) / 2) * 9 * mid_ch_calc ** 2 + (s + 1) * mid_ch_calc * ch_out
			# groups = max(round((total_calc / param_compute) ** 0.5), 1)
			if l_i == 0:
				dense_layer = SimpleDenseLayer(channels[l_i], mid_chs[l_i], channels[l_i + 1], depths[l_i], layer_group[l_i])
			else:
				dense_layer = SimpleDenseLayer(channels[l_i], mid_chs[l_i], channels[l_i + 1], depths[l_i], layer_group[l_i],out_groups=layer_group[l_i])

			self.ch_out = dense_layer.out_ch
			self.layers[f'{l_i}'] = dense_layer

		self.layer_count = layers

	def forward(self, x):
		for i, layer in enumerate(self.layers.values()):
			x = layer(x)
			if i < self.layer_count - 1:
				x = self.downsample_input(x)
		return x


def dumb_calc(s, j=2):
	return ((j ** 2 * (s + 1) ** 2 + 2 * j * (81 * s ** 2 + 82 * s + 1) + 1) ** 0.5 - j * (s + 1) - 1) / (
				9 * s * (s + 1))

	return ((328 * s ** 2 + 336 * s + 9) ** 0.5 - 2 * s - 3) / (9 * s * (s + 1))


class Decoder(nn.Module):
	def __init__(self, ch_in, ch_out, layers=4) -> None:
		super().__init__()

		self.layers = layers
		ratio = (ch_out / ch_in) ** (1 / layers)
		channels = [round((ch_in * ratio ** i)) for i in range(layers + 1)]
		AppLog.info(f'Decoder channels {channels}')
		param_compute = channels[-2] * channels[-1] * 4

		decoder_layers = ModuleDict()
		for layer in range(layers):
			ch_in = channels[layer]
			ch_out = channels[layer + 1]

			s = (layers - layer) + 2
			mid_ch_calc = round(dumb_calc(s,ratio) * ch_in)
			total_calc = ch_in * mid_ch_calc + (s * (s + 1) / 2) * 9 * mid_ch_calc ** 2 + (s + 1) * mid_ch_calc * ch_out
			groups = max(round((total_calc / param_compute) ** 0.5), 1)
			kernel_list = [3 for _ in range(s)]

			dec_layer = SimpleDenseLayer(mid_ch_calc, ch_out * 4, groups, kernel_list)
			decoder_layers[f'{layer}'] = dec_layer

		self.decoder_layers = decoder_layers
		self.size = [512, 512]
		self.last_activation = nn.Tanh()

		self.last_compress = nn.LazyConv2d(out_channels=3, kernel_size=1, padding=0)
		self.upsample2 = nn.PixelShuffle(2)

	def set_size(self, h, w):
		self.size = [h, w]

	def forward(self, latent_z):
		[h, w] = self.size

		z_m = latent_z


		for i, dec_layer in enumerate(self.decoder_layers.values()):
			upscaled = self.upsample2(z_m)
			z_m = dec_layer(upscaled)

		z_m = interpolate(z_m, size=[h, w], mode='nearest-exact')
		compressed_output = self.last_compress(z_m)
		return self.last_activation(compressed_output)


class ImageCodec(nn.Module):
	def __init__(self, enc_chin, latent_channels, dec_chout, enc_layers=4, dec_layers=4):
		super().__init__()

		self.encoder = Encoder(enc_chin, latent_channels, layers=enc_layers)
		self.decoder = Decoder(latent_channels, dec_chout, layers=dec_layers)

	def forward(self, x):
		nh, nw = torch.ceil(x.shape[-2] / 16) * 16, torch.ceil(x.shape[-1] / 16) * 16
		x = interpolate(x, size=[nh, nw], mode='nearest-exact')
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
	model_res, latent = model(data)
	final_res = scale_decoder_data(model_res)
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
	# in_ch = 161
	# out_ch = 256
	#
	# m = calcParamsForMid(in_ch, out_ch, [[1, 3, 5] for _ in range(3)], 25 * in_ch * out_ch)
	# mid_ch = round(m)
	# AppLog.info(f'Encoder middle channel : {m} , {mid_ch}')
	#
	#
	# stack_layer = CodecMultiKernelStack(in_ch, mid_ch, out_ch, (3, [1, 3, 5]))
	#
	# AppLog.info(f'Stacked Encoder : {stack_layer}')
	# summary(stack_layer, [(12, in_ch, 40, 40)])
	# poolS = PoolSelector(127)
	# summary(poolS, [(12, 256, 40, 40)])

	# For stacked.
	# m (c k_1 + c + 2 k_c + 2 o + 1) + m^2 (2 k_s + l - 1) + o

	# chn =[64, 128, 192, 256]
	# chn.reverse()
	# dec = Encoder(chn)
	# enc = Encoder(64, 256, 6, 1 / 16)
	enc = ImageCodec(64, 256, 48)
	AppLog.info(f'Encoder : {enc}')
	summary(enc, [(12, 3, 256, 256)])
