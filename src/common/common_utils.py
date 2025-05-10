import inspect
import logging
import os
import sys
from dataclasses import dataclass
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from math import log2
from pathlib import Path
from queue import Queue
from typing import Optional, Tuple

import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from numpy import log2
from torch import Tensor, nn
from torch.nn import Conv2d
from torchvision.io import decode_image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2.functional import crop_image, normalize, rotate, to_dtype


# This is for logging applications
class AppLog:
	_instance: Optional['AppLog'] = None
	_logger: Optional[logging.Logger] = None

	def __new__(cls):
		if cls._instance is None:
			cls._instance = super(AppLog, cls).__new__(cls)
			cls._instance._initialize_logger()
		return cls._instance

	def _initialize_logger(self) -> None:
		"""Initialize the logger with rotating file handler"""
		if self._logger is None:
			pid = os.getpid()
			self._logger = logging.getLogger('ApplicationLogger')
			self._logger.setLevel(logging.INFO)  # Default level

			log_que = Queue(maxsize=1024)
			q_handle = QueueHandler(log_que)
			file_dir = Path(__file__).parent.parent.resolve()
			logdir = file_dir.parent / 'log'
			logfile = logdir / f'application_{pid}.log'
			print('Logging to {}'.format(logfile))

			handler = RotatingFileHandler(logfile,
										  maxBytes=1 * 1024 * 1024,  # 1MB
										  backupCount=5, encoding='utf-8')

			# Format for the log messages
			formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
			handler.setFormatter(formatter)

			# Console handler
			console_handler = logging.StreamHandler(sys.stdout)
			console_handler.setFormatter(formatter)
			console_handler.setLevel(logging.INFO)

			# self._logger.addHandler(console_handler)
			self._q_listener = QueueListener(log_que, handler, console_handler)
			self._logger.addHandler(q_handle)
			self._q_listener.start()

	@classmethod
	def set_level(cls, level: str) -> None:
		"""Set the logging level"""
		if cls._instance is None:
			cls._instance = cls()

		level_map = {'DEBUG'   : logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING,
					 'ERROR'   : logging.ERROR,
					 'CRITICAL': logging.CRITICAL}

		cls._instance._logger.setLevel(level_map.get(level.upper(), logging.INFO))

	@classmethod
	def shut_down(cls) -> None:
		"""Shuts down the logger"""
		if cls._instance is not None:
			try:
				cls._instance._q_listener.stop()
			except AttributeError:
				pass

	def _log(self, level: str, message: str) -> None:
		"""Internal method to handle logging with caller information"""
		# Get caller frame information
		caller_frame = inspect.currentframe().f_back.f_back
		filename = os.path.basename(caller_frame.f_code.co_filename)
		func_name = caller_frame.f_code.co_name

		# Combine filename, function name and message
		full_message = f"[{filename}:{func_name}] {message}"

		if level == 'DEBUG':
			self._logger.debug(full_message)
		elif level == 'INFO':
			self._logger.info(full_message)
		elif level == 'WARNING':
			self._logger.warning(full_message)
		elif level == 'ERROR':
			self._logger.error(full_message)
		elif level == 'CRITICAL':
			self._logger.critical(full_message)

	@classmethod
	def debug(cls, message: str) -> None:
		if cls._instance is None:
			cls._instance = cls()
		cls._instance._log('DEBUG', message)

	@classmethod
	def info(cls, message: str) -> None:
		if cls._instance is None:
			cls._instance = cls()
		cls._instance._log('INFO', message)

	@classmethod
	def warning(cls, message: str) -> None:
		if cls._instance is None:
			cls._instance = cls()
		cls._instance._log('WARNING', message)

	@classmethod
	def error(cls, message: str) -> None:
		if cls._instance is None:
			cls._instance = cls()
		cls._instance._log('ERROR', message)

	@classmethod
	def critical(cls, message: str) -> None:
		if cls._instance is None:
			cls._instance = cls()
		cls._instance._log('CRITICAL', message)


def show_image(img_tensor):
	(c, h, w) = img_tensor.shape
	min_ch_val = torch.amin(img_tensor, dim=[1, 2])
	max_ch_val = torch.amax(img_tensor, dim=[1, 2])
	ch_range = max_ch_val - min_ch_val
	normal = normalize(img_tensor, min_ch_val.tolist(), ch_range.tolist())
	num_image = torch.permute(normal, (1, 2, 0)).numpy()
	plt.imshow(num_image)
	plt.show()


def convert_rgb_to_complex(image):
	transform_matrix = torch.tensor([[0.299, 0.587, 0.114],
									 [-0.168736, -0.331264, 0.5],
									 [0.5, -0.418688, -0.081312]], dtype=torch.float32)
	# Reshape the image to (N, C, H, W) if it's not already
	if image.ndim == 3:
		image = image.unsqueeze(0)
	# Permute the image to (N, H, W, C)
	image = image.permute(0, 2, 3, 1)
	ycbcr_image = torch.trunc(torch.tensordot(image.to(dtype=torch.float32), transform_matrix, dims=([3], [1])))
	# Permute the image to (C, N, H, W)
	# ycbcr_image_ch_form = ycbcr_image.permute(3, 0, 1, 2)
	luma_only = ycbcr_image[:, :, :, 0]
	cb_cr_only = ycbcr_image[:, :, :, 1:]
	complex_cb_cr = torch.view_as_complex(cb_cr_only.contiguous())
	print(f'luma = {luma_only}')
	print(f'cb_cr = {complex_cb_cr}')
	hue_angle = complex_cb_cr.angle()
	real = torch.trunc(luma_only * torch.cos(hue_angle))
	imag = torch.trunc(luma_only * torch.sin(hue_angle))
	complex_image = real + 1j * imag
	return complex_image


def convert_complex_to_rgb(image):
	transform_matrix = torch.tensor([[0.299, 0.587, 0.114],
									 [-0.168736, -0.331264, 0.5],
									 [0.5, -0.418688, -0.081312]], dtype=torch.float32)
	inverse_transform_matrix = torch.inverse(transform_matrix)
	# Reshape the image to (N, H, W, C) if it's not already
	if image.ndim == 3:
		image = image.unsqueeze(0)
	image_luma = image.abs()
	print(f' luma_max: {image_luma.max()} , luma_min: {image_luma.min()} ')
	image_hue_angle = image.angle()
	cb = torch.cos(image_hue_angle) * (1 - image_luma / 255) * image_luma / (1.772 * 2)
	cr = torch.sin(image_hue_angle) * (1 - image_luma / 255) * image_luma / (1.402 * 2)
	print(f'cb_max = {cb.max()} , cb_min = {cb.min()} , cr = {cr.max()} , cr = {cr.min()}')

	ycbcr_image = torch.cat((image_luma, cb, cr), dim=1)
	# Permute to (N, H, W, C)
	ycbcr_image = ycbcr_image.permute(0, 2, 3, 1)
	rgb_image_raw = torch.tensordot(ycbcr_image, inverse_transform_matrix, dims=([3], [1]))

	# Permute the image to (N,C,H,W)
	rgb_image = rgb_image_raw.permute(0, 3, 1, 2)

	print(f' r_max = {rgb_image[:, 0, :, :].max()} , r_min = {rgb_image[:, 0, :, :].min()} ')
	print(f' g_max = {rgb_image[:, 1, :, :].max()} , g_min = {rgb_image[:, 1, :, :].min()} ')
	print(f' b_max = {rgb_image[:, 2, :, :].max()} , b_min = {rgb_image[:, 2, :, :].min()} ')

	return rgb_image


class CNNUtils:
	@staticmethod
	def calculate_coverage(k_size: int, cnn_layers: int, downsampling_factor: float):
		k = k_size
		m = cnn_layers - 1
		r = downsampling_factor

		coverage = (2 - 2 * k + r ** m * (r * k + k - 2)) / (r - 1)
		return coverage

	@staticmethod
	def calculate_train_params(in_ch: int, out_ch: int, k_size: int, sep_param_ratio=1):
		r = sep_param_ratio
		t = out_ch / in_ch
		k = k_size
		if sep_param_ratio == 1:
			return in_ch * out_ch * k_size ** 2
		if t != 1:
			a = log2((t + 1) / (t * r * k)) / log2(1 / t)
			intermediate_channels = round(in_ch ** (1 - a) * out_ch ** a)
			new_a = log2(intermediate_channels / in_ch) / log2(out_ch / in_ch)
			min_frac = (t + 1) * t ** (-1) / k
			max_frac = (t + 1) * t ** 0 / k
			print(f'The intermediate channel = {intermediate_channels}, where a = {new_a:.2f}')
			print(f'The min frac = {min_frac}, max frac = {max_frac} for {sep_param_ratio}')
			return intermediate_channels * k * (in_ch + out_ch)
		else:
			print(f'The ratio should be {2 / k} if not already')
			return in_ch * out_ch * k_size ** 2 * sep_param_ratio

	@staticmethod
	def calculate_total_cnn_params(k_sizes: list[int] | int | Tuple[int, float], channels: list[int]):
		in_channels = channels[:-1]
		out_channels = channels[1:]
		match k_sizes:
			case int(_):
				in_out_pairs = zip(in_channels, out_channels)
				return sum([CNNUtils.calculate_train_params(in_ch, out_ch, k_sizes) for in_ch, out_ch in in_out_pairs])
			case list(_):
				in_out_pairs_ksizes = zip(in_channels, out_channels, k_sizes)
				return sum([CNNUtils.calculate_train_params(in_ch, out_ch, k_size) for in_ch, out_ch, k_size in
							in_out_pairs_ksizes])
			case tuple(_):
				in_out_pairs_ksizes_ratio = zip(in_channels, out_channels)
				return sum([CNNUtils.calculate_train_params(in_ch, out_ch, k_sizes[0], k_sizes[1]) for in_ch, out_ch in
							in_out_pairs_ksizes_ratio])
		return None


def get_diffs(prepared_image):
	img_45 = rotate(prepared_image, 45, interpolation=InterpolationMode.BILINEAR, expand=True)
	(_, h45, w45) = img_45.shape
	(_, h, w) = prepared_image.shape
	top, left = (h45 - h) // 2, (w45 - w) // 2
	diff_45_w = torch.diff(img_45, dim=-1)
	diff_45_h = torch.diff(img_45, dim=-2)
	diff_rotback_w = rotate(diff_45_w, -45, interpolation=InterpolationMode.BILINEAR, expand=False)
	diff_rotback_h = rotate(diff_45_h, -45, interpolation=InterpolationMode.BILINEAR, expand=False)
	diff_rotback_w_cropped = crop_image(diff_rotback_w, top, left, h, w)
	diff_rotback_h_cropped = crop_image(diff_rotback_h, top, left, h, w)
	diff_w = diff_rotback_w_cropped - diff_rotback_h_cropped
	diff_h = diff_rotback_w_cropped + diff_rotback_h_cropped
	return diff_h[:, 1:, 1:], diff_w[:, 1:, 1:]


@dataclass
class Ratio:
	ratio: float


@dataclass
class ParamA:
	param_a: float


@dataclass
class IntermediateChannel:
	intermediate_channel: int


InterChannelType = IntermediateChannel | ParamA | Ratio


def generate_separated_kernels(input_channel: int, output_channel: int, k_size: int, inter_ch: InterChannelType,
							   add_padding: bool = True, bias: bool = False, stride: int = 1, switch=True) -> tuple[
	Conv2d, Conv2d]:
	c_in = input_channel
	c_out = output_channel
	t = c_out / c_in
	k = k_size

	c_intermediate = min(c_in, c_out)

	if inter_ch is not None:
		match inter_ch:
			case IntermediateChannel(intermediate_channel=int(ic)):
				c_intermediate = ic
			case ParamA(param_a=float(a)):
				c_intermediate = round((c_in ** (1 - a) * c_out ** a))
			case Ratio(ratio=float(r)):
				a = log2((t + 1) / (t * r * k)) / log2(1 / t)
				c_intermediate = round((c_in ** (1 - a) * c_out ** a))

	AppLog.info(f'c_in={c_in}, c_intermediate={c_intermediate}, c_out={c_out}')

	padding = k // 2 if add_padding else 0
	if switch:
		conv_layer_1 = nn.Conv2d(c_in, c_intermediate, (k, 1), padding=(padding, 0), bias=bias, stride=(stride, 1))
		conv_layer_2 = nn.Conv2d(c_intermediate, c_out, (1, k), padding=(0, padding), bias=bias, stride=(1, stride))
	else:
		conv_layer_1 = nn.Conv2d(c_in, c_intermediate, (1, k), padding=(0, padding), bias=bias, stride=(1, stride))
		conv_layer_2 = nn.Conv2d(c_intermediate, c_out, (k, 1), padding=(padding, 0), bias=bias, stride=(stride, 1))

	return conv_layer_1, conv_layer_2


def acquire_image(image_path):
	return to_dtype(decode_image(image_path, mode='RGB'), dtype=torch.float32, scale=True)


# Squashes a C,H,W tensor to 1,H,W tensor
def squash_n_to_1(tensor: Tensor):
	abs_val = torch.abs(tensor)
	min_val = torch.amin(abs_val, dim=[-3, -2, -1])
	max_val = torch.amax(abs_val, dim=[-3, -2, -1])
	normed = torch.div((abs_val - min_val), (max_val - min_val))

	ch_one = torch.ones_like(normed[:, 0, :, :])
	numer = ch_one.clone()
	for c in range(tensor.shape[-3]):
		ch = normed[:, c, :, :]
		numer = torch.mul(ch_one - ch, numer)
	denom = torch.zeros_like(numer)
	for c in range(tensor.shape[-3]):
		ch = normed[:, c, :, :]
		denom_obj = torch.div(numer, ch_one - ch)
		denom = torch.add(denom, torch.mul(denom_obj, ch))
	prepared_tensor = torch.div(numer, denom)
	prepared_tensor[prepared_tensor != prepared_tensor] = 0

	return prepared_tensor


def visualize_tensor(tensor: Tensor):
	imshow()


def quincunx_diff_avg(img):
	top_left = img[..., :-1, :-1]
	top_right = img[..., :-1, 1:]
	bottom_left = img[..., 1:, :-1]
	bottom_right = img[..., 1:, 1:]

	img_diff_w = ((top_right - bottom_left) + (bottom_right - top_left)) / 2
	img_diff_h = (-(top_right - bottom_left) + (bottom_right - top_left)) / 2
	img_avg = (top_right + bottom_left + bottom_right + top_left) / 4
	return img_diff_w, img_diff_h, img_avg
