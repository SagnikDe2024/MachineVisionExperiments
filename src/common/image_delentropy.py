from math import lcm

import numpy as np
import torch
from numpy import dtype, ndarray
from torch import nn
from torch.nn import ModuleDict
from torchvision.io import write_png

from src.common.common_utils import AppLog
from src.common.common_utils import convert_complex_to_rgb


def calculate_pdf(pic: torch.tensor):
	print(f'shape: {pic.shape}')
	x_shift_kernel = torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=torch.half)
	x_shift_kernel = x_shift_kernel.repeat(3, 1, 1)
	x_shift_kernel = x_shift_kernel.unsqueeze(0)
	# x_shift_kernel.repeat(4,3)
	y_shift_kernel = torch.tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.half).unsqueeze(0).unsqueeze(0)
	pic_un = pic.unsqueeze(0).to(dtype=torch.half)
	(ch, h, w) = pic.shape
	pic_uny = pic_un.permute(0, 1, 3, 2)

	pic_shift_x = torch.concat((torch.zeros(ch, h, 1), pic[:, :, 1:]), 2)
	pic_shift_y = torch.concat((torch.zeros(ch, 1, w), pic[:, 1:, :]), 1)
	pic_shift_z = torch.concat((torch.zeros(1, h, w), pic[1:, :, :]), 0)
	shft_x = pic - pic_shift_x
	shft_y = pic - pic_shift_y
	shft_z = pic - pic_shift_z

	# pic_x = fn.conv2d(pic_un, x_shift_kernel, bias=None, padding=1)
	# # pic_y = fn.conv2d(pic_un, y_shift_kernel, bias=None, padding=1)
	# pic_y_un = fn.conv2d(pic_uny, x_shift_kernel, bias=None, padding=1)
	# pic_y = pic_y_un.permute(0, 1, 3, 2)

	if ch > 1:
		# pic_unz = pic_un.permute(0, 3, 2, 1)
		# pic_z_un = fn.conv2d(pic_unz, x_shift_kernel, bias=None, padding=1)
		# pic_z = pic_z_un.permute(0, 3, 2, 1)
		# pic_x = pic_x.squeeze()
		# pic_y = pic_y.squeeze()
		# pic_z = pic_z.squeeze()
		pic_x = shft_x
		pic_y = shft_y
		pic_z = shft_z
		pdf: ndarray[tuple[int, int, int], dtype[np.float32]] = np.zeros([511, 511, 511], dtype=np.float32)
		for px in range(w):
			for py in range(h):
				for pz in range(ch):
					xv = int(pic_x[pz, py, px])
					yv = int(pic_y[pz, py, px])
					zv = int(pic_z[pz, py, px])
					pdf[zv + 255, yv + 255, xv + 255] += 1
		pdf = pdf / np.sum(pdf)
		return pdf
	else:
		# pic_x = pic_x.squeeze().squeeze()
		# pic_y = pic_y.squeeze().squeeze()
		pdf: ndarray[tuple[int, int], dtype[np.float32]] = np.zeros([511, 511], dtype=np.float32)
		# for px in range(w):
		#     for py in range(h):
		#         xv = int(pic_x[py, px])
		#         yv = int(pic_y[py, px])
		#         pdf[yv + 255, xv + 255] += 1
		pdf = pdf / np.sum(pdf)
		return pdf


# def get_gradients(pic : torch.tensor):
#     print(f'shape: {pic.shape}')
#     x_shift_kernel = torch.tensor([[0, 0, 0], [-1/2, 1, -1/2], [0, 0, 0]], dtype=torch.half)

def find_complex_diff(pic: torch.tensor):
	(n, h, w) = pic.shape
	xdiff = torch.diff(pic, dim=2, prepend=torch.zeros(n, h, 1))

	write_png(convert_complex_to_rgb(xdiff).squeeze(0).to(dtype=torch.uint8), "../../out/complex_xdiff.png")
	ydiff = torch.diff(pic, dim=1, prepend=torch.zeros(n, 1, w))
	write_png(convert_complex_to_rgb(ydiff).squeeze(0).to(dtype=torch.uint8), "../../out/complex_ydiff.png")

	npx = np.array(xdiff)
	npy = np.array(ydiff)
	print(f'{pic}')
	print(f'npx: {npx}, npy: {npy}')
	r_diff = torch.where(xdiff.real.abs() > ydiff.imag.abs(), xdiff.real, ydiff.imag)
	i_diff = torch.where(xdiff.imag.abs() > ydiff.real.abs(), xdiff.imag, ydiff.real)
	complex_diff = r_diff + 1j * i_diff
	return complex_diff


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
		self.inp_conv = nn.LazyConv2d(out_channels=in_ch, kernel_size=1, padding=0, bias=True, groups=in_groups)
		self.mid_conv_modules = ModuleDict()
		use_bias = not normed
		cost = 0
		kernel_list = [3 for _ in range(depth)]
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
		self.out_conv = nn.LazyConv2d(out_channels=out_ch, kernel_size=1, padding=0, bias=True, groups=out_groups)
		cost += out_ch * 2 + mid_ch * len(kernel_list)
		AppLog.info(
			f'Dense layer in_ch={self.in_ch} mid_ch={self.mid_ch}, out_ch={self.out_ch}, groups={groups}, '
			f'kernels={kernel_list}')
		AppLog.info(f'Approx params {cost}')
		AppLog.info('------------------------------------')

	def forward(self, x):
		dense_input = self.inp_conv(x)

		for mid_conv in self.mid_conv_modules.values():
			mid_conv_res = mid_conv(dense_input)
			dense_input = torch.cat([mid_conv_res, dense_input], dim=1)
		out = self.out_conv(dense_input)
		return out
