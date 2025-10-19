from math import ceil, lcm

import torch
from torch import nn
from torch.nn import ModuleDict
from torchinfo import summary
from torchvision.transforms.v2.functional import center_crop, pad

from src.common.common_utils import AppLog


class SimpleDenseLayer(nn.Module):

	def __init__(self, in_ch, mid_ch, out_ch, depth, groups, normed=True, in_groups=1, out_groups=1, dropped_out=0.2):
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
			shuffle_1 = nn.ChannelShuffle(2)
			shuffle_2 = nn.ChannelShuffle(groups)
			dropped_out_layer = nn.Dropout(dropped_out)
			act = nn.Mish()
			self.mid_conv_modules[f'{k}'] = nn.Sequential(shuffle_1, conv, shuffle_2, norm, act, dropped_out_layer)
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

def fill_as_req(img,size_mul=16):
	h,w = img.shape[-2:]
	h_new = ceil(h / size_mul) * size_mul
	w_new = ceil(w / size_mul) * size_mul
	left = (w_new - w) // 2
	right = w_new - w - left
	bottom = (h_new - h) // 2
	top = h_new - h - bottom
	img_new = pad(img, [left,top,right,bottom],padding_mode='symmetric')
	return img_new,h,w

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

def calc_channels_depth_and_midchs(in_ch, out_ch, in_depth, out_depth, layers, mid_ch_st=12):
	ch_ratio = (out_ch / in_ch) ** (1 / (layers-1))
	channels = [ round(in_ch*ch_ratio**i/4)*4 for i in range(-1,layers)]
	depth_ratio = (out_depth / in_depth) ** (1 / (layers-1))
	depths = [ round(in_depth*depth_ratio**i) for i in range(layers)]
	mid_ch_r = ch_ratio/depth_ratio
	mid_chs = [ round(mid_ch_st*mid_ch_r**i/4)*4  for i in range(layers)]
	groups = [ mch // 4 for mch in mid_chs ]
	return channels, depths, mid_chs, groups


class Encoder(nn.Module):
	def __init__(self, ch_in_enc, ch_out_enc, layers=5, out_groups=16, min_depth=9, max_depth=18, min_mid_ch=12):
		super().__init__()
		self.min_depth = min_depth
		self.max_depth = max_depth
		channels, depths, mid_chs, layer_group = calc_channels_depth_and_midchs(ch_in_enc, ch_out_enc, self.min_depth,
		                                                                        self.max_depth, layers, mid_ch_st=min_mid_ch)
		AppLog.info(f'Encoder channels {channels}, depths {depths}, mid_chs {mid_chs}, groups {layer_group}')


		# self.downsample_input = nn.PixelUnshuffle(2)
		all_layers = []
		for l_i in range(layers):
			# total_calc = ch_in * mid_ch_calc + (s * (s + 1) / 2) * 9 * mid_ch_calc ** 2 + (s + 1) * mid_ch_calc * ch_out
			# groups = max(round((total_calc / param_compute) ** 0.5), 1)
			if l_i == 0:
				dense_layer = SimpleDenseLayer(channels[l_i], mid_chs[l_i], channels[l_i + 1], depths[l_i], layer_group[l_i])
			elif l_i < layers - 1:
				dense_layer = SimpleDenseLayer(channels[l_i], mid_chs[l_i], channels[l_i + 1], depths[l_i], layer_group[l_i],out_groups=layer_group[l_i])
			else:
				dense_layer = SimpleDenseLayer(channels[l_i], mid_chs[l_i], channels[l_i + 1], depths[l_i], out_groups ,in_groups=4, out_groups=out_groups)
			all_layers.append(dense_layer)
			if l_i < layers - 1:
				all_layers.append(nn.PixelUnshuffle(2))

			self.ch_out = dense_layer.out_ch
		self.final_norm = nn.GroupNorm(1, self.ch_out)
		self.final_act = nn.Tanh()
		all_layers.append(self.final_norm)
		all_layers.append(self.final_act)
		self.all_layers_list = all_layers

		self.encoder_layers = nn.Sequential(*self.all_layers_list)
		self.layer_count = layers

	def forward(self, x, ratio=1.0):
		xf,h,w = fill_as_req(x)
		encoded = self.encoder_layers(xf)
		encoded_channels = encoded.shape[1]
		zeroed = round((encoded_channels-1) * (1-ratio))
		new_encode = encoded[:,zeroed:,:,:]
		return new_encode , h, w


def dumb_calc(s, j=2):
	return ((j ** 2 * (s + 1) ** 2 + 2 * j * (81 * s ** 2 + 82 * s + 1) + 1) ** 0.5 - j * (s + 1) - 1) / (
				9 * s * (s + 1))

	return ((328 * s ** 2 + 336 * s + 9) ** 0.5 - 2 * s - 3) / (9 * s * (s + 1))


class Decoder(nn.Module):
	def __init__(self, ch_in_dec, ch_out_dec, layers=5, in_group=32, min_depth=8, max_depth=9, min_mid_ch=12) -> None:
		super().__init__()
		self.min_depth = min_depth
		self.max_depth = max_depth
		self.min_mid_ch = min_mid_ch
		channels, depths, mid_chs, layer_group = calc_channels_depth_and_midchs(ch_out_dec, ch_in_dec, self.min_depth,
		                                                                        self.max_depth, layers, mid_ch_st=self.min_mid_ch)
		channels.reverse()
		depths.reverse()
		mid_chs.reverse()
		layer_group.reverse()
		self.layers = layers

		AppLog.info(f'Decoder channels {channels}, depths {depths}, mid_chs {mid_chs}, groups {layer_group}')

		all_layers = []
		self.input_layers = -1
		for layer in range(layers):
			ch_in = channels[layer]
			ch_out = channels[layer + 1]
			s = depths[layer]
			mid_ch_calc = mid_chs[layer]
			# total_calc = ch_in * mid_ch_calc + (s * (s + 1) / 2) * 9 * mid_ch_calc ** 2 + (s + 1) * mid_ch_calc *
			# ch_out
			groups = layer_group[layer]
			if layer == 0:
				dec_layer = SimpleDenseLayer(ch_in, mid_ch_calc, ch_out, s, groups, in_groups=in_group,
				                             out_groups=groups)
				self.input_layers = dec_layer.in_ch
			else:
				dec_layer = SimpleDenseLayer(ch_in, mid_ch_calc, ch_out, s, groups, out_groups=groups)
			all_layers.append(dec_layer)
			if layer < layers - 1:
				upscale = nn.PixelShuffle(2)
				all_layers.append(upscale)

		self.last_compress = nn.LazyConv2d(out_channels=3, kernel_size=1, padding=0)

		self.last_activation = nn.Tanh()
		all_layers.append(self.last_compress)
		all_layers.append(self.last_activation)
		self.decoder_layers = nn.Sequential(*all_layers)

	def forward(self, latent_z, h, w):
		n, c, lh, lw = latent_z.shape
		zeros = torch.zeros([n, self.input_layers - c, lh, lw], device=latent_z.device)
		latent_z = torch.cat([zeros, latent_z], dim=1)

		out_uncropped = self.decoder_layers(latent_z)

		out = center_crop(out_uncropped, output_size=[h, w])
		return out


class ImageCodec(nn.Module):
	def __init__(self, enc_chin, latent_channels, dec_chout, enc_layers=5, dec_layers=5):
		super().__init__()

		self.encoder = Encoder(enc_chin, latent_channels, layers=enc_layers)
		self.decoder = Decoder(latent_channels, dec_chout, layers=dec_layers)

	def forward(self, x, ratio=1.0):
		latent, h, w = self.encoder.forward(x, ratio=ratio)

		final_res = self.decoder(latent, h, w)
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
	enc = ImageCodec(64, 768, 48)
	AppLog.info(f'Encoder : {enc}')
	inp = torch.randn(16, 3, 320, 320)
	ratio = 1.0
	summary(enc, input_data=(inp, ratio))
