import random
from pathlib import Path

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import Tensor
from torchvision.transforms.v2.functional import resize

from src.common import common_utils


def add_gaussian_noise(image: Tensor, mean: float = 0.0, std: float = 0.1) -> Tensor:
	"""
	Add Gaussian noise to an image using PyTorch.
	
	Args:
		image: Input image as torch tensor
		mean: Mean of Gaussian distribution (default: 0.0)
		std: Standard deviation of Gaussian distribution (default: 0.1)
		
	Returns:
		Noisy image as torch tensor
	"""
	noise = torch.normal(mean=mean, std=std, size=image.shape, device=image.device)
	noisy_image = image + noise
	return torch.clamp(noisy_image, 0.0, 1.0)


def add_stuck_pixels(image: Tensor, density: float = 0.01) -> Tensor:
	"""
	Add stuck/dead pixels to an image using PyTorch.
	
	Args:
		image: Input image as torch tensor
		density: Percentage of pixels to be stuck (default: 0.01 = 1%)
		
	Returns:
		Image with stuck pixels as torch tensor
	"""
	defective_image = image.clone()
	total_pixels = image.shape[-2] * image.shape[-1]
	num_stuck = int(total_pixels * density)

	for _ in range(num_stuck):
		# Select random position
		h = random.randint(0, image.shape[-2] - 1)
		w = random.randint(0, image.shape[-1] - 1)
		# Set pixel to either black (0) or white (1)
		value = random.choice([0.0, 1.0])
		defective_image[..., h, w] = value

	return defective_image


def add_photon_shot_noise(image: Tensor, scaling_factor: float = 100.0) -> Tensor:
	"""
	Add photon shot noise to an image using PyTorch's Poisson distribution.
	The noise is more prominent in darker regions of the image.
	
	Args:
		image: Input image as torch tensor (can be batched)
		scaling_factor: Controls the noise intensity (default: 100.0)
		
	Returns:
		Image with photon shot noise as torch tensor
	"""
	# Scale image for Poisson noise simulation
	scaled_img = image * scaling_factor

	# Generate Poisson noise using PyTorch
	noisy_img = torch.poisson(scaled_img) / scaling_factor

	return torch.clamp(noisy_img, 0.0, 1.0)


def add_chromatic_aberration(image: Tensor, shift: float = 2.0) -> Tensor:
	"""
	Add chromatic aberration effect by shifting RGB channels.

	Args:
		image: Input image as torch tensor (N,C,H,W)
		shift: Amount of pixel shift for color channels (default: 2.0)

	Returns:
		Image with chromatic aberration effect
	"""
	# Split into channels
	r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]

	# Shift red channel left and blue channel right
	r_shifted = F.pad(r, (-int(shift), int(shift), 0, 0))
	b_shifted = F.pad(b, (int(shift), -int(shift), 0, 0))

	# Combine channels
	return torch.cat([r_shifted, g, b_shifted], dim=1)


def add_grid_flare(image: Tensor, intensity: float = 0.3, frequency: float = 20.0) -> Tensor:
	"""
	Add grid flare effect with periodic patterns.

	Args:
		image: Input image as torch tensor
		intensity: Strength of the effect (default: 0.3)
		frequency: Frequency of grid pattern (default: 20.0)

	Returns:
		Image with grid flare effect
	"""
	h, w = image.shape[-2:]
	y = torch.linspace(0, h - 1, h)
	x = torch.linspace(0, w - 1, w)

	# Create grid pattern
	xx, yy = torch.meshgrid(x, y, indexing='xy')
	pattern = torch.sin(xx / frequency) * torch.sin(yy / frequency)
	pattern = pattern.to(image.device)

	# Add pattern to all channels
	flare = pattern.expand_as(image) * intensity
	return torch.clamp(image + flare, 0.0, 1.0)


def add_lens_flare(image: Tensor, num_flares: int = 5, intensity: float = 0.5) -> Tensor:
	"""
	Add lens flare effect with multiple light sources.

	Args:
		image: Input image as torch tensor
		num_flares: Number of flare points (default: 5)
		intensity: Strength of the flare effect (default: 0.5)

	Returns:
		Image with lens flare effect
	"""
	h, w = image.shape[-2:]
	flare_img = torch.zeros_like(image)

	for _ in range(num_flares):
		# Random flare position
		x = random.randint(0, w - 1)
		y = random.randint(0, h - 1)

		# Create radial gradient
		yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
		yy = yy.to(image.device)
		xx = xx.to(image.device)

		dist = torch.sqrt((xx - x) ** 2 + (yy - y) ** 2)
		flare = 1.0 / (1.0 + dist / 10.0)
		print(f'flare : {flare.shape} , image : {image.shape}')
		flare = flare.unsqueeze(0).expand_as(image)

		flare_img += flare * intensity / num_flares

	return torch.clamp(image + flare_img, 0.0, 1.0)


def add_coma(image: Tensor, strength: float = 0.2) -> Tensor:
	"""
	Simulate coma aberration effect. Coma aberration is an optical aberration where
	off-axis points appear comet-shaped. The effect increases with distance from the optical axis.

	Args:
		image: Input image as torch tensor
		strength: Strength of the coma effect (default: 0.2)

	Returns:
		Image with coma aberration effect
	"""
	h, w = image.shape[-2:]

	# Create normalized coordinates
	y, x = torch.meshgrid(torch.linspace(-1.0, 1.0, h), torch.linspace(-1.0, 1.0, w), indexing='ij')

	# Calculate polar coordinates
	r = torch.sqrt(x ** 2 + y ** 2)
	theta = torch.atan2(y, x)

	# Coma aberration distortion
	# The r³cosθ term creates the characteristic comet-like shape
	x_displacement = strength * (r ** 3 * torch.cos(theta))
	y_displacement = strength * (r ** 3 * torch.sin(theta))

	# Create sampling grid
	grid_x = torch.clamp(x + x_displacement, -1, 1)
	grid_y = torch.clamp(y + y_displacement, -1, 1)

	# Stack into sampling grid
	grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).to(image.device)

	# Sample with bilinear interpolation
	return F.grid_sample(image, grid, mode='bilinear', padding_mode='reflection', align_corners=False)


def add_posterization(image: Tensor, levels: int = 5) -> Tensor:
	"""
	Apply posterization effect by reducing color levels.

	Args:
		image: Input image as torch tensor
		levels: Number of color levels (default: 5)

	Returns:
		Posterized image
	"""
	factor = (levels - 1) / 1.0
	return torch.round(image * factor) / factor


def add_jpeg_artifacts(image: Tensor, quality: int = 50) -> Tensor:
	"""
	Simulate JPEG compression artifacts using DCT transform.

	Args:
		image: Input image as torch tensor (N,C,H,W)
		quality: JPEG quality factor (0-100, lower means more compression artifacts)

	Returns:
		Image with JPEG compression artifacts
	"""
	# Ensure dimensions are multiples of 8
	h, w = image.shape[-2:]
	h_pad = (8 - h % 8) % 8
	w_pad = (8 - w % 8) % 8
	image = F.pad(image, (0, w_pad, 0, h_pad))

	# Prepare quantization matrix
	q = torch.tensor(
			[[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],
					[14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77],
					[24, 35, 55, 64, 81, 104, 113, 92], [49, 64, 78, 87, 103, 121, 120, 101],
					[72, 92, 95, 98, 112, 100, 103, 99]], device=image.device).float()

	# Scale quantization matrix based on quality
	scale = (100 - quality) / 50
	q = torch.clamp(q * scale, 1, 255)

	# Process image in 8x8 blocks
	# blocks = image.unfold(2, 8, 8).unfold(3, 8, 8)
	blocks = image.contiguous().view(-1, 8, 8)

	# Apply DCT using FFT
	dct_blocks = torch.fft.fft2(torch.fft.fft2(blocks, dim=1), dim=2).real

	# Quantize
	quantized = torch.round(dct_blocks / q)
	dequantized = quantized * q

	# Inverse DCT using IFFT
	idct_blocks = torch.fft.ifft2(torch.fft.ifft2(dequantized, dim=2), dim=1).real
	print(
		f'Size of blocks: {blocks.shape}, dct_blocks : {dct_blocks.shape}, idct_blocks : {idct_blocks.shape}, '
		f'image : {image.shape}')

	# Reshape back

	output = idct_blocks.view(image.shape[0], image.shape[1], image.shape[2] // 8, 8, image.shape[3] // 8, 8)
	output = output.permute(0, 1, 2, 4, 3, 5).contiguous()
	output = output.view(image.shape)

	# Remove padding
	if h_pad > 0 or w_pad > 0:
		output = output[..., :h, :w]

	return torch.clamp(output, 0.0, 1.0)


def tests():
	current_dir = Path(__file__)  # Get the directory of current file
	image_path = current_dir.parent.parent.parent / 'data' / 'normal_pic.jpg'

	# Load and pre-process the image
	image = common_utils.acquire_image(image_path)
	image = image.unsqueeze(0)  # Add batch dimension

	# Resize the image
	max_dim = max(image.shape[2], image.shape[3])
	resize_factor = 1024.0 / max_dim
	new_size = [int(image.shape[2] * resize_factor), int(image.shape[3] * resize_factor)]

	resized_image = resize(image, new_size)

	def show_image(img_tensor):
		img_tensor = img_tensor.squeeze(0)
		plt.imshow(img_tensor.permute(1, 2, 0))
		plt.show()

	# Apply each defect and display the result
	# noisy_image = add_gaussian_noise(resized_image)
	# show_image(noisy_image)
	#
	# stuck_pixels_image = add_stuck_pixels(resized_image)
	# show_image(stuck_pixels_image)
	#
	# shot_noise_image = add_photon_shot_noise(resized_image)
	# show_image(shot_noise_image)
	#
	# chromatic_aberration_image = add_chromatic_aberration(resized_image)
	# show_image(chromatic_aberration_image)

	# grid_flare_image = add_grid_flare(resized_image)
	# show_image(grid_flare_image)

	# lens_flare_image = add_lens_flare(resized_image)
	# show_image(lens_flare_image)

	# coma_image = add_coma(resized_image)
	# show_image(coma_image)

	# posterization_image = add_posterization(resized_image)
	# show_image(posterization_image)

	jpeg_artifacts_image = add_jpeg_artifacts(resized_image)
	show_image(jpeg_artifacts_image)


if __name__ == '__main__':
	tests()
