import numpy as np
import torch
from torch import nn
from torch.nn.functional import conv2d, interpolate, mse_loss
from torchmetrics.functional.image import visual_information_fidelity

from src.common.common_utils import AppLog, quincunx_diff_avg

def get_gradient_weights():
	diffxf1 = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
	diffxd1 = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
	diffyf1 = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	diffyd1 = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
	diffxf = torch.zeros(3, 3, 3, 3)
	diffyf = torch.zeros(3, 3, 3, 3)
	diffxd = torch.zeros(3, 3, 3, 3)
	diffyd = torch.zeros(3, 3, 3, 3)
	for i in range(3):
		diffxf[i, i] = diffxf1
		diffyf[i, i] = diffyf1
		diffxd[i, i] = diffxd1
		diffyd[i, i] = diffyd1

	gradient_convs = [diffxf, diffyf, diffxd, diffyd]
	return gradient_convs


class MultiScaleGradientLoss(nn.Module):
	def __init__(self, max_downsample=4, steps_to_downsample=4):
		super().__init__()
		self.dummy_param = nn.Parameter(torch.empty(0))
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
		self.gradient_convs = get_gradient_weights()



	def get_weighted_gradient_loss(self, scale, weight, inferred_image, target_image):
		target_gradients = self.get_gradients(scale, target_image)
		inferred_gradients = self.get_gradients(scale, inferred_image)
		inferred_and_target_gradients = zip(inferred_gradients, target_gradients)
		mse_losses = map(lambda inf_target_grad : self.loss2(inf_target_grad[0],inf_target_grad[1]) , inferred_and_target_gradients)
		return sum(mse_losses) * weight


	def forward(self, inferred_image, target_image):
		sc_w_fn= lambda sc,w : self.get_weighted_gradient_loss(sc,w,inferred_image,target_image)
		losses = [sc_w_fn(sc,w) for sc,w in zip(self.loss_scales,self.loss_weights)  ]
		return sum(losses)


def funky_reduce(t1, t2):
	t_r = (t1 + t2 - 2*t1*t2)/(1-t1*t2)
	return torch.nan_to_num(t_r, nan=1, posinf=1, neginf=1)


def get_saturation(image):
	min_max = torch.aminmax(image, dim=1, keepdim=True)
	inv_sat  = torch.nan_to_num(min_max.min / min_max.max, nan=1, posinf=1, neginf=1)
	return 1-inv_sat





class MultiscalePerceptualLoss(nn.Module):
	def __init__(self, max_downsample=8, steps_to_downsample=4):
		super().__init__()
		self.dummy_param = nn.Parameter(torch.empty(0))
		self.luminosity = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
		self.loss1 = nn.L1Loss()
		self.max_downsample = max_downsample
		self.steps_to_downsample = steps_to_downsample

		diffxd, diffxf, diffyd, diffyf = get_gradient_weights()

		self.convs = [diffxf / 4,diffyf/4,diffxd/4,diffyd/4]

		self.downsample_ratio = max_downsample ** (-1 / (steps_to_downsample - 1))
		AppLog.info(f"Downsample ratio: {self.downsample_ratio}")

	def get_lumniosity(self, image):
		this_dev = self.dummy_param.device
		self.luminosity = self.luminosity.to(this_dev)
		return torch.sum(image * self.luminosity, dim=1, keepdim=True)

	def get_colour_diff(self,image):
		img_diffs = map(lambda conv: conv2d(image, conv.to(self.dummy_param.device), padding=1),self.convs)
		colour_diffs = list(map(lambda img_diff: torch.pow(self.del_c(image, img_diff),1/2),img_diffs))

		r1_res = funky_reduce(colour_diffs[0],colour_diffs[1])
		r2_res = funky_reduce(colour_diffs[2],colour_diffs[3])
		all_res = funky_reduce(r1_res,r2_res)
		return all_res

		#
		# max_diffs = torch.max(torch.concat(list(colour_diffs),dim=1),dim=1,keepdim=True)[0]
		# return max_diffs

	def get_color_difference(self, image):
		this_dev = self.dummy_param.device
		img_diff_x = conv2d(image, self.conv_diffxf.to(this_dev), padding=1)
		img_diff_y = conv2d(image, self.conv_diffyf.to(this_dev), padding=1)
		AppLog.info(f'image_diff_x {img_diff_x.abs().max()}, image_diff_y {img_diff_y.abs().max()}')
		c_delx = torch.pow(self.del_c(image, img_diff_x),1/2)
		c_dely = torch.pow(self.del_c(image, img_diff_y),1/2)
		AppLog.info(f'c_delx_max {c_delx.abs().max()}, c_dely_max {c_dely.abs().max()}')
		c_del = (c_delx + c_dely - 2*c_delx*c_dely)/(1-c_delx*c_dely)
		return torch.nan_to_num(c_del, nan=1, posinf=1, neginf=1)

	def del_c(self, image, img_diff):
		red = image[:, 0:1, :, :]
		dr = img_diff[:, 0:1, :, :]
		dg = img_diff[:, 1:2, :, :]
		db = img_diff[:, 2:3, :, :]
		c_delta2 = ((2 + red) * dr * dr + 4 * dg * dg + (2 + 255 / 256 - red) * db * db)/9
		AppLog.info(f'c_delta {c_delta2.shape}, max delta = {c_delta2.max()}')
		return c_delta2

	def weighted_pixel_imp(self, image):
		# inv_lum = 1-self.get_lumniosity(image)

		# sat = self.get_saturation(image)
		sat = 0
		# inv_lum = torch.zeros_like(sat)
		inv_lum = 0
		c_del = self.get_colour_diff(image)
		mul1 = inv_lum * sat
		mul2 = sat * c_del
		mul3 = c_del * inv_lum
		mulall = inv_lum * sat * c_del
		num = inv_lum + sat + c_del - 2*(mul1 + mul2 + mul3) + 3*mulall
		denom = 1 - mul1 - mul2 - mul3 + 2*mulall
		weight = num / denom
		return torch.nan_to_num(weight, nan=1, posinf=1, neginf=1)


	def get_gradients_no_scale(self, sc_image):
		this_dev = self.dummy_param.device
		img_diff_x = conv2d(sc_image, self.conv_diffxf.to(this_dev), padding=1)
		img_diff_y = conv2d(sc_image, self.conv_diffyf.to(this_dev), padding=1)
		img_avg = conv2d(sc_image, self.conv2d_avg.to(this_dev), padding=1)
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
		self.dummy_param = nn.Parameter(torch.empty(0))
		self.loss2 = nn.MSELoss()
		diffxf1 = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
		diffxd1 = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
		diffyf1 = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
		diffyd1 = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
		diffxf = torch.zeros(3,3,3,3)
		diffyf = torch.zeros(3,3,3,3)
		diffxd = torch.zeros(3,3,3,3)
		diffyd = torch.zeros(3,3,3,3)
		for i in range(3):
			diffxf[i,i] = diffxf1
			diffyf[i,i] = diffyf1
			diffxd[i,i] = diffxd1
			diffyd[i,i] = diffyd1

		self.convs = [diffxf / 4,diffyf/4,diffxd/4,diffyd/4]

	def del_c(self, image, img_diff):
		red = image[:, 0:1, :, :]
		dr = img_diff[:, 0:1, :, :]
		dg = img_diff[:, 1:2, :, :]
		db = img_diff[:, 2:3, :, :]
		c_delta2 = ((2 + red) * dr * dr + 4 * dg * dg + (2 + 255 / 256 - red) * db * db)/9
		return c_delta2

	def get_colour_diff(self,image):
		img_diffs = map(lambda conv: conv2d(image, conv.to(self.dummy_param.device), padding=1),self.convs)
		colour_diffs = list(map(lambda img_diff: torch.pow(self.del_c(image, img_diff),1/2),img_diffs))

		r1_res = funky_reduce(colour_diffs[0],colour_diffs[1])
		r2_res = funky_reduce(colour_diffs[2],colour_diffs[3])
		all_res = funky_reduce(r1_res,r2_res)
		return all_res

	def forward(self, inferred_image, target_image):
		all_res = self.get_colour_diff(target_image) + get_saturation(target_image)
		new_weight = all_res + 1
		weight = torch.concat([new_weight,new_weight,new_weight],dim=1)
		ls = mse_loss(inferred_image, target_image,weight=weight)
		return ls
		# return self.loss2(inferred_image, target_image,new_weight)


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
