import numpy as np
import torch
from torch import nn
from torch.nn.functional import interpolate
from torchmetrics.image import VisualInformationFidelity

from src.common.common_utils import AppLog, quincunx_diff_avg


class MultiScaleGradientLoss(nn.Module):
	def __init__(self, max_downsample=8, steps_to_downsample=4):
		super().__init__()
		md = max_downsample
		steps = steps_to_downsample
		self.max_downsample = md
		self.steps_to_downsample = steps_to_downsample
		self.loss_scales = [md ** (step / (steps - 1)) for step in range(steps)]
		loss_weights_r = [np.exp(-step) for step in range(steps)]
		loss_weight_sum = sum(loss_weights_r)
		self.loss_weights = [loss_weight / loss_weight_sum for loss_weight in loss_weights_r]
		AppLog.info(f"Loss scales: {self.loss_scales}")
		self.loss2 = nn.MSELoss()
		self.loss1 = nn.L1Loss()

	def get_gradients(self, scale, img):
		sc_image = interpolate(img, scale_factor=scale, mode='bilinear', align_corners=False)
		img_diff_w, img_diff_h, _ = quincunx_diff_avg(sc_image)
		return img_diff_w, img_diff_h

	def forward(self, inferred_image, target_image):
		losses = []
		for scale_factor, loss_weight in zip(self.loss_scales, self.loss_weights):
			inf_diff_w_weighted, inf_diff_h_weighted = self.get_gradients(scale_factor, inferred_image)
			target_diff_w_weighted, target_diff_h_weighted = self.get_gradients(scale_factor, target_image)
			total_loss: nn.MSELoss = loss_weight * (
					self.loss2(inf_diff_w_weighted, target_diff_w_weighted) + self.loss2(inf_diff_h_weighted,
																						 target_diff_h_weighted))
			losses.append(total_loss)
		return sum(losses)


class MultiscalePerceptualLoss(nn.Module):
	def __init__(self, max_downsample=8, steps_to_downsample=4):
		super().__init__()
		self.luminosity = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
		self.loss1 = nn.L1Loss()
		self.max_downsample = max_downsample
		self.steps_to_downsample = steps_to_downsample
		self.loss_scales = [max_downsample ** (step / (steps_to_downsample - 1)) for step in range(
				steps_to_downsample)]
		AppLog.info(f"Loss scales: {self.loss_scales}")

	def get_lumniosity(self, image):
		return torch.sum(image * self.luminosity, dim=1, keepdim=True)

	def get_gradients(self, scale, img):
		sc_image = interpolate(img, scale_factor=scale, mode='bilinear', align_corners=False)
		img_diff_x, img_diff_y, img_avg = quincunx_diff_avg(sc_image)
		return img_diff_x, img_diff_y, img_avg

	def forward(self, inferred_image, target_image):
		inferred_image_lumniosity = self.get_lumniosity(inferred_image)
		target_image_lumniosity = self.get_lumniosity(target_image)
		losses = []

		for scale_factor in self.loss_scales:
			inf_diff_x, inf_diff_y, _ = self.get_gradients(scale_factor, inferred_image_lumniosity)
			target_diff_x, target_diff_y, target_avg = self.get_gradients(scale_factor, target_image_lumniosity)
			resp_avg = 1 / (target_avg + 1e-5)
			total_loss = self.loss1(inf_diff_x * resp_avg, target_diff_x * resp_avg) + self.loss1(inf_diff_y *
																								  resp_avg,
																								  target_diff_y *
																								  resp_avg)
			losses.append(total_loss)
		return sum(losses)


class ReconstructionLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.loss1 = nn.L1Loss()

	def forward(self, inferred_image, target_image):
		return self.loss1(inferred_image, target_image)


class VisualInformationFidelityLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.vif_metric = VisualInformationFidelity()
		AppLog.info("Initialized VisualInformationFidelityLoss")

	def forward(self, inferred_image, target_image):
		# VIF is a similarity metric (higher is better)
		# For a loss function, we want lower to be better, so we use 1 - VIF
		vif_score = self.vif_metric(inferred_image, target_image)
		# Ensure the score is between 0 and 1
		vif_score = torch.clamp(vif_score, 0.0, 1.0)
		return 1.0 - vif_score
