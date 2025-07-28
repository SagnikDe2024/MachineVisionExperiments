from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import Conv2d, ModuleDict
from torch.nn.functional import interpolate
from torchinfo import summary

from src.common.common_utils import AppLog


# class L1BatchNorm2D(nn.Module):
# 	def __init__(self, channels):
# 		super().__init__()
# 		self.bnl1_weights = nn.Parameter(torch.ones(channels))
# 		self.bnl1_bias = nn.Parameter(torch.zeros(channels))
# 		self.bnl1_const = math.pi/2
#
# 	def forward(self, x):
#
# 		l1_mean = torch.mean(x, dim=-4, keepdim=True)
# 		diff = x - l1_mean
# 		l1_std = torch.abs(diff).mean(dim=-4, keepdim=True)
# 		x_normed = diff / (self.bnl1_const*l1_std + 1e-8)
# 		x_normed_weights_bias = x_normed * self.bnl1_weights + self.bnl1_bias
# 		return x_normed_weights_bias


class EncoderLayer(nn.Module):
	def __init__(self, input_ch, mid_ch, output_ch, pool_params, stack: list[tuple[int, int]] | tuple[int, list[int]]):
		super().__init__()
		self.cnn_stack = CodecMultiKernelStack(input_ch, mid_ch, output_ch, stack)
		self.poolingAvg = nn.AvgPool2d(kernel_size=2)
		self.poolingMax = nn.MaxPool2d(kernel_size=2)
		self.poolSelector = PoolSelector(pool_params)

	def forward(self, x: Tensor):
		x_res = self.cnn_stack(x)
		pooledAvg = self.poolingAvg(x_res)
		pooledMax = self.poolingMax(x_res)
		selected = self.poolSelector(x_res)
		selected_dim = selected.view(selected.shape[0], selected.shape[1], 1, 1)
		pooled = pooledAvg * selected_dim + pooledMax * (1 - selected_dim)
		return pooled


class DecoderLayer(nn.Module):
	def __init__(self, input_ch, mid_ch, output_ch, stack: list[tuple[int, int]] | tuple[int, list[int]],
	             upsample: Optional[int] = 2,last=False):
		super().__init__()
		self.cnn_stack = CodecMultiKernelStack(input_ch, mid_ch, output_ch, stack) if not last else CodecMultiKernelStack(input_ch, mid_ch, mid_ch, stack[:-1])
		self.upscale = nn.Upsample(scale_factor=upsample, mode='bicubic') if upsample > 1 else None
		self.last_layer = nn.Sequential(nn.Conv2d(mid_ch, output_ch, kernel_size=1),nn.Tanh()) if last else nn.Identity()
		self.h = 128
		self.w = 128

	def set_h_w(self, h, w):
		self.h = h
		self.w = w

	def forward(self, x: Tensor):
		if self.upscale is not None:
			upsampled = self.upscale(x)
		else:
			upsampled: Tensor = interpolate(x, size=(self.h, self.w), mode='bicubic')
		cnn_res = self.cnn_stack(upsampled)
		return self.last_layer(cnn_res)


def create_sep_kernels(input_channels, output_channels, kernel_size):
	min_channels = min(input_channels, output_channels)
	padding = kernel_size // 2
	conv1 = nn.Conv2d(in_channels=input_channels, out_channels=min_channels, kernel_size=(1, kernel_size),
			padding=(0, padding), bias=False)
	conv2 = nn.Conv2d(in_channels=min_channels, out_channels=output_channels, kernel_size=(kernel_size, 1),
			padding=(padding, 0), bias=False)
	return conv1, conv2



class Encoder1stLayer(nn.Module):
	def __init__(self, input_channels,output_channels, pool_params, kernels=1):
		super().__init__()
		mid_channels = output_channels // 2
		kernel_list = [(2*k + 1) for k in range(1,kernels+1)]
		self.front_layer = nn.Sequential(nn.Conv2d(input_channels, mid_channels, 3,padding=1,bias=False),nn.BatchNorm2d(mid_channels),nn.Mish())
		self.passthrough = nn.Conv2d(input_channels, mid_channels, kernel_size=1)
		self.second_layer = CodecMultiKernelBlock(mid_channels,mid_channels, output_channels, kernel_list)
		self.poolingAvg = nn.AvgPool2d(kernel_size=2)
		self.poolingMax = nn.MaxPool2d(kernel_size=2)
		self.poolSelector = PoolSelector(pool_params)

	def forward(self, x):
		activate = self.front_layer(x)
		passthrough = self.passthrough(x)
		x_res = self.second_layer(activate + passthrough)
		selected = self.poolSelector(x_res)
		selected_dim = selected.view(selected.shape[0], selected.shape[1], 1, 1)
		pooledAvg = self.poolingAvg(x_res)
		pooledMax = self.poolingMax(x_res)
		pooled = pooledAvg * selected_dim + pooledMax * (1 - selected_dim)
		return pooled


class ResLinearBlock(nn.Module):
	def __init__(self, in_ch, out_ch, last=False):
		super().__init__()
		self.linear = nn.Linear(in_ch, out_ch, bias=False)
		self.bn = nn.BatchNorm1d(out_ch)
		self.act = nn.Mish() if not last else nn.Sigmoid()
		self.passthrough = nn.AdaptiveAvgPool1d(out_ch)
		self.last = last

	def forward(self, x):
		x_p = self.passthrough(x)
		x_l = self.linear(x)
		x_bn = self.bn(x_l)
		x_act = self.act(x_bn)
		with_res = x_act + x_p if not self.last else x_act
		return with_res


class PoolSelector(nn.Module):
	def __init__(self, exp_params):
		super().__init__()
		cube_vol = (exp_params * 1.5 + 1) ** 0.5
		cube_side = int(cube_vol ** (1 / 3))
		self.avg_pool = nn.AdaptiveAvgPool2d(cube_side)
		self.conv_down = nn.LazyConv2d(cube_side, kernel_size=1, padding=0, bias=True)
		new_size = cube_side ** 3
		self.linear_part = nn.Sequential()
		AppLog.info(f'The size is {new_size}')
		while new_size >= 2:
			down_size = new_size // 2
			if down_size == 1:
				self.linear_part.append(ResLinearBlock(new_size, down_size, last=True))
				break
			else:
				self.linear_part.append(ResLinearBlock(new_size, down_size, last=False))
			new_size = down_size
		self.all_ops = nn.Sequential(self.avg_pool, self.conv_down, nn.Flatten(), self.linear_part)

	def forward(self, x):
		return self.all_ops(x)


class CodecSubBlockDepthSep(nn.Module):
	def __init__(self, in_ch, mid_ch, out_ch, kernel_size):
		super().__init__()
		self.input_compress = nn.Conv2d(in_ch, mid_ch, kernel_size=1, padding=0,
				bias=False) if in_ch != mid_ch else nn.Identity()
		conv1, conv2 = create_sep_kernels(mid_ch, mid_ch, kernel_size)
		self.output_compress = nn.Conv2d(mid_ch, out_ch, kernel_size=1, padding=0)
		self.active_path = nn.Sequential(self.input_compress, conv1, conv2, nn.BatchNorm2d(mid_ch), nn.Mish(),
				self.output_compress)

	def forward(self, x):
		return self.active_path(x)

class CodecSubBlockFull(nn.Module):
	def __init__(self, in_ch, mid_ch, out_ch, kernel_size):
		super().__init__()
		self.input_compress = nn.Conv2d(in_ch, mid_ch, kernel_size=1, padding=0,
				bias=False) if in_ch != mid_ch else nn.Identity()
		self.conv = Conv2d(mid_ch, mid_ch, kernel_size=kernel_size, padding=kernel_size // 2,bias=False)
		self.output_compress = nn.Conv2d(mid_ch, out_ch, kernel_size=1, padding=0)
		self.active_path = nn.Sequential(self.input_compress, self.conv, nn.BatchNorm2d(mid_ch), nn.Mish(),self.output_compress)

	def forward(self, x):
		return self.active_path(x)

class CodecMultiKernelBlock(nn.Module):
	def __init__(self, in_channels, middle_ch, out_channels, kernels: list[int]):
		super().__init__()
		n = len(kernels)
		AppLog.info(f'CodecMultiKernelBlock: i={in_channels} m={middle_ch} o={out_channels} ks={kernels} ')
		# This is to ensure that the sum of output channels is exactly the same as the number of output_channels
		out_each_ch = [round(i * out_channels / n) for i in range(0, n + 1)]
		self.passthrough = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
		self.kernels = ModuleDict()
		for i, k_s in enumerate(kernels):
			out_each = out_each_ch[i + 1] - out_each_ch[i]
			self.kernels[f'k{i}'] = CodecSubBlockDepthSep(in_channels, middle_ch, out_each, k_s)

	def forward(self, x):
		evaluated = []
		passthrough = self.passthrough(x)
		for act in self.kernels.values():
			activated = act(x)
			evaluated.append(activated)
		concat_activated = torch.cat(evaluated, dim=1)
		return concat_activated + passthrough


def calc_stack(stack):
	_stack = [[k[0] + 2 * y for y in range(k[1])] for k in stack]
	return _stack


class CodecMultiKernelStack(nn.Module):
	def __init__(self, input_ch, mid_ch, output_ch, stack: list[tuple[int, int]] | tuple[int, list[int]]):
		super().__init__()
		AppLog.info(f'stack {stack}')
		if type(stack) is list(tuple[int, int]) or type(stack) is list:
			_stack = calc_stack(stack)
		else:
			kst = stack[1]
			stacks = stack[0]
			_stack = [[*kst] for _ in range(stacks)]
		stacks = len(_stack)

		self.kernel_stack = nn.Sequential()
		if stacks == 1:
			enc_block = CodecMultiKernelBlock(input_ch, output_ch, mid_ch, _stack[0])
			self.kernel_stack.append(enc_block)
		else:
			for i, ks in enumerate(_stack):
				if i == 0:
					enc_block = CodecMultiKernelBlock(input_ch, mid_ch, mid_ch, ks)
				elif i == stacks - 1:
					enc_block = CodecMultiKernelBlock(mid_ch, mid_ch, output_ch, ks)
				else:
					enc_block = CodecMultiKernelBlock(mid_ch, mid_ch, mid_ch, ks)
				self.kernel_stack.append(enc_block)

	def forward(self, x):
		x_res = self.kernel_stack(x)
		return x_res




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


class SimpleEncoder4Layer(nn.Module):
	def __init__(self, input_ch, output_ch, kernels_per_stack, mid_ch_layer_inc, stack_layer_ramp_inc, kernel_stack_layer_inc, pool_params):
		super().__init__()
		calculated_ramp_values =  [(input_ch*((output_ch/input_ch) ** (i/3)), mid_ch_layer_inc(i/3), stack_layer_ramp_inc(i/3)) for i in range(4)]
		rounded = list(map(lambda x: [round(x[0]),round(x[1]),round(x[2])] , calculated_ramp_values))
		channels, mid_channels, stacks = tuple(map(list,zip(*rounded)))
		ks = kernels_per_stack
		AppLog.info(f'Encoder4Layer: ch={channels} ks={ks} m={mid_channels} s={stacks} ')
		kernel_bases_layer_stack = [ [ round(kernel_stack_layer_inc(s/(stacks[l]-1), l/3))*2 + 1  for s in range(stacks[l]) ] for l in range(4) ]
		for l in range(4):
			AppLog.info(f'Kernel bases for layer {l+1} are {kernel_bases_layer_stack[l]}')

		self.encoder_layers = nn.Sequential()
		for i in range(4):
			if i == 0:
				enc = Encoder1stLayer(3, channels[i], pool_params, ks)
				self.encoder_layers.append(enc)
				continue
			else:
				kernel_ranges = list(map(lambda x : (x,ks),  kernel_bases_layer_stack[i]))
				enc = EncoderLayer(channels[i-1], mid_channels[i], channels[i], pool_params, kernel_ranges)
				self.encoder_layers.append(enc)

	def forward(self, x):
		x_res = self.encoder_layers(x)
		return x_res


class SimpleDecoder4Layer(nn.Module):
	def __init__(self, input_ch, output_ch, kernels_per_stack, mid_ch_layer_dec, stack_layer_dec, kernel_stack_layer_dec):
		super().__init__()
		calculated_ramp_values = [
				(input_ch*((output_ch / input_ch) ** (i / 3)), mid_ch_layer_dec(i/3), stack_layer_dec(i/3)) for i in
				range(0, 4)]
		rounded = list(map(lambda x: [round(x[0]), round(x[1]), round(x[2])], calculated_ramp_values))
		channels, mid_channels, stacks = tuple(map(list,zip(*rounded)))
		ks = kernels_per_stack
		AppLog.info(f'Decoder4Layer: ch={channels} ks={ks} m={mid_channels} s={stacks} ')
		kernel_bases_layer_stack = [ [ round(kernel_stack_layer_dec(s/(stacks[l]-1), l/3))*2 + 1   for s in range(stacks[l]) ] for l in range(4) ]
		for l in range(4):
			AppLog.info(f'Kernel bases for layer {l+1} are {kernel_bases_layer_stack[l]}')

		self.decoder_layers = nn.Sequential()
		for i in range(4):
			kernel_ranges = list(map(lambda x: (x, ks), kernel_bases_layer_stack[i]))
			if i == 3:
				dec = DecoderLayer(channels[i], mid_channels[i], 3, kernel_ranges, last=True)
				self.decoder_layers.append(dec)
				continue
			dec = DecoderLayer(channels[i], mid_channels[i], channels[i+1], kernel_ranges)
			self.decoder_layers.append(dec)

	def forward(self, x):
		x_res = self.decoder_layers(x)
		# x_act = self.last_layer_act(x_res)
		return x_res

class SimpleCodec(nn.Module):
	def __init__(self, encoder, decoder):
		super().__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self, x):
		x_enc = self.encoder(x)
		x_dec = self.decoder(x_enc)
		return x_dec,x_enc

if __name__ == '__main__':

	enc = SimpleEncoder4Layer(55, 256, 2, lambda l: 32+16*l , lambda l: 2+l*2*3, lambda s,l: 1 + s*l*1, 512)
	dec = SimpleDecoder4Layer(256, 48, 3, lambda l: 24+16*(1-l), lambda l: 2+(1-l)*3*1.5, lambda s,l: 0)
	# decl = DecoderLayer(256,48,140, (3,[1,3,5]))
	codec = SimpleCodec(enc, dec)
	# print(de)
	summary(codec, (12,3, 256, 256))
