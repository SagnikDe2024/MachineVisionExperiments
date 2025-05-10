import numpy as np
import torch
from torch import nn
from torch.nn.functional import interpolate
from src.wip.find_diff import quincunx_diff_avg
from src.common.common_utils import Applog


class MultiScaleGradientLoss(nn.Module):
	def __init__(self, max_downsample, steps_to_downsample):
		super().__init__()
		md = max_downsample
		steps = steps_to_downsample
		self.max_downsample = md
		self.steps_to_downsample = steps_to_downsample
		self.loss_scales = [md ** (step / (steps - 1)) for step in range(steps)]
		loss_weights_r = [np.exp(-step) for step in range(steps)]
		loss_weight_sum = sum(loss_weights_r)
		self.loss_weights = [loss_weight / loss_weight_sum for loss_weight in loss_weights_r]
		Applog.info(f"Loss scales: {self.loss_scales}")
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
	def __init__(self):
		super().__init__()
		self.luminosity = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)

	def get_lumniosity(self, image):
		return torch.sum(image * self.luminosity, dim=1, keepdim=True)

	def forward(self, inferred_image, target_image):
		inferred_image_lumniosity = self.get_lumniosity(inferred_image)
		target_image_lumniosity = self.get_lumniosity(target_image)
		inf_diff_x, inf_diff_y, _ = quincunx_diff_avg(inferred_image_lumniosity)
		target_diff_x, target_diff_y, target_avg = quincunx_diff_avg(target_image_lumniosity)
		resp_avg = 1 / (target_avg + 1e-5)
		return self.loss1(inf_diff_x * resp_avg, target_diff_x * resp_avg) + self.loss1(inf_diff_y * resp_avg,
																						target_diff_y * resp_avg)


class ReconstructionLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.loss1 = nn.L1Loss()

	def forward(self, inferred_image, target_image):
		return self.loss1(inferred_image, target_image)
