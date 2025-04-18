import inspect
import logging
import os
import sys
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from queue import Queue
from typing import Optional

import torch
from matplotlib import pyplot as plt
from torchvision.transforms.v2.functional import normalize


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
			file_dir = Path(__file__).parent.resolve()
			logdir = file_dir.parent.parent / 'log'
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
