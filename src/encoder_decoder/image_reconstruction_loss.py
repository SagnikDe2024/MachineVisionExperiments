import numpy as np
import torch
from torch import nn
from torch.nn.functional import conv2d, interpolate
from torchmetrics.functional.image import visual_information_fidelity

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
		self.dummy_param = nn.Parameter(torch.empty(0))
		self.luminosity = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
		self.loss1 = nn.L1Loss()
		self.max_downsample = max_downsample
		self.steps_to_downsample = steps_to_downsample
		diffx1 = torch.tensor([[-1, -1], [1, 1]])
		diffy1 = torch.tensor([[-1, 1], [-1, 1]])
		avg1 = torch.tensor([[1, 1], [1, 1]])
		diffx3 = torch.stack([diffx1, diffx1, diffx1], dim=0)
		diffy3 = torch.stack([diffy1, diffy1, diffy1], dim=0)
		avg3 = torch.stack([avg1, avg1, avg1], dim=0)
		self.conv2d_diffx = diffx3.unsqueeze(0) / 2
		self.conv2d_diffy = diffy3.unsqueeze(0) / 2
		self.conv2d_avg = avg3.unsqueeze(0) / 4
		self.downsample_ratio = max_downsample ** (-1 / (steps_to_downsample - 1))
		AppLog.info(f"Downsample ratio: {self.downsample_ratio}")
		AppLog.info(
			f'Stacked shapes diffx: {self.conv2d_diffx.shape}, diffy: {self.conv2d_diffy.shape}, '
			f'avg: {self.conv2d_avg.shape}')

	def get_lumniosity(self, image):
		this_dev = self.dummy_param.device
		self.luminosity = self.luminosity.to(this_dev)
		return torch.sum(image * self.luminosity, dim=1, keepdim=True)

	def get_gradients_no_scale(self, sc_image):
		this_dev = self.dummy_param.device
		img_diff_x = conv2d(sc_image, self.conv2d_diffx.to(this_dev), padding=0)
		img_diff_y = conv2d(sc_image, self.conv2d_diffy.to(this_dev), padding=0)
		img_avg = conv2d(sc_image, self.conv2d_avg.to(this_dev), padding=0)
		return img_diff_x, img_diff_y, img_avg

	def calc_loss(self, inferred_image, target_image):
		inf_diff_x, inf_diff_y, _ = self.get_gradients_no_scale(inferred_image)
		target_diff_x, target_diff_y, target_avg = self.get_gradients_no_scale(target_image)
		avg_lum = self.get_lumniosity(target_avg)
		resp_avg = 1 / (avg_lum + 1e-5)
		total_loss = self.loss1(inf_diff_x * resp_avg, target_diff_x * resp_avg) + self.loss1(inf_diff_y *
		                                                                                      resp_avg,
		                                                                                      target_diff_y *
		                                                                                      resp_avg)
		return total_loss

	def forward(self, inferred_image, target_image):
		loss = self.calc_loss(inferred_image, target_image)

		for steps in range(1, self.steps_to_downsample):
			inferred_image = interpolate(inferred_image, scale_factor=self.downsample_ratio,
			                                        mode='bilinear')
			target_image = interpolate(target_image, scale_factor=self.downsample_ratio,
			                                      mode='bilinear')
			newloss = self.calc_loss(inferred_image, target_image)
			loss+=newloss
		return loss


class ReconstructionLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.loss1 = nn.L1Loss()

	def forward(self, inferred_image, target_image):
		return self.loss1(inferred_image, target_image)


class ReconstructionLossRelative(nn.Module):
	def __init__(self):
		super().__init__()

		self.loss2 = nn.MSELoss()
		self.dummy_param = nn.Parameter(torch.empty(0))
		self.luminosity = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)

	def forward(self, inferred_image, target_image):
		weight = self.get_weight(target_image)
		weight = (weight * 3 + 1) / 4
		return self.loss2(inferred_image * weight, target_image * weight)

	def get_weight(self, target_image):
		this_dev = self.dummy_param.device
		self.luminosity = self.luminosity.to(this_dev)
		min_max = torch.aminmax(target_image, dim=1, keepdim=True)
		inv_sat = min_max.min / min_max.max
		res = torch.nan_to_num(inv_sat, nan=1, posinf=1, neginf=1)
		lum = torch.sum(target_image * self.luminosity, dim=1, keepdim=True)
		num = res + lum
		denom = res * lum + res + lum
		weight = num / denom
		weight = torch.nan_to_num(weight, nan=1, posinf=1, neginf=1)
		return weight


class VisualInformationFidelityLoss(nn.Module):
	def __init__(self):
		super().__init__()
		# self.vif_metric = VisualInformationFidelity()
		AppLog.info("Initialized VisualInformationFidelityLoss")

	def forward(self, inferred_image, target_image):
		# VIF is a similarity metric (higher is better)
		# For a loss function, we want lower to be better, so we use 1 - VIF
		vif_score = visual_information_fidelity(inferred_image,target_image)
		# Ensure the score is between 0 and 1
		return 1.0 - vif_score
