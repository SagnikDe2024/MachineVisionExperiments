import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2.functional import crop_image, rgb_to_grayscale_image, rotate_image
from torchvision.utils import save_image

from src.common.common_utils import acquire_image


def find_central_diff(img):
	img_diff_w = (img[..., 2:] - img[..., :-2]) / 2
	img_diff_h = (img[..., 2:, :] - img[..., :-2, :]) / 2
	return img_diff_w, img_diff_h


def rot_diff(img):
	shape = img.shape
	h, w = shape[-2:]
	rot_45 = rotate_image(img, 45, interpolation=InterpolationMode.BILINEAR, expand=True)
	shape_rot = rot_45.shape
	h_rot, w_rot = shape_rot[-2:]
	r_dw, r_dh = find_central_diff(rot_45)
	un_rot_w_raw = rotate_image(r_dw, -45, interpolation=InterpolationMode.BILINEAR, expand=False)
	un_rot_h_raw = rotate_image(r_dh, -45, interpolation=InterpolationMode.BILINEAR, expand=False)
	top = (h_rot - h) // 2
	left = (w_rot - w) // 2
	un_rot_w = crop_image(un_rot_w_raw, top + 1, left, h - 2, w - 2)
	un_rot_h = crop_image(un_rot_h_raw, top, left + 1, h - 2, w - 2)
	diff_h = un_rot_h + un_rot_w
	diff_w = un_rot_w - un_rot_h
	return diff_w, diff_h


def relative_diff(img):
	diff_w, diff_h = rot_diff(img)
	crop_1_pixel = crop_image(img, 1, 1, img.shape[-2] - 2, img.shape[-1] - 2)
	epsilon = 1 / (2 ** 16 - 1)
	rel_diff_w = diff_w / (crop_1_pixel + epsilon)
	rel_diff_h = diff_h / (crop_1_pixel + epsilon)
	return rel_diff_w, rel_diff_h


def false_colour(img_diff):
	max_value = img_diff.max()
	mean_v = img_diff.mean()
	print(f'max_value = {max_value}, mean_v = {mean_v}')
	max_2 = max_value / 2
	max_4 = max_value / 4
	max_34 = max_value * (3 / 4)
	zero_img = torch.zeros_like(img_diff)
	ones = torch.ones_like(img_diff)
	bl_img = torch.where(img_diff <= max_2 * ones, 4 * (1 - img_diff / max_2) * (img_diff / max_2), 0)
	red_img = torch.where(torch.logical_and(max_4 * ones < img_diff, img_diff <= max_34 * ones),
						  4 * (1 - (img_diff - max_4) / max_2) * ((img_diff - max_4) / max_2), 0)
	yl_img = torch.where(img_diff <= max_value * ones,
						 4 * (1 - (img_diff - max_2) / max_2) * ((img_diff - max_2) / max_2), 0)
	red_blue_concat = torch.cat((red_img, zero_img, bl_img), dim=0)
	yellow_concat = torch.cat((yl_img, yl_img, zero_img), dim=0)
	result = torch.clamp(red_blue_concat + yellow_concat, 0, 1)  # red_blue_concat + yellow_concat
	return max_2, max_4, max_value, result


if __name__ == "__main__":
	img_acq = acquire_image('data/normal_pic.jpg')
	gray = rgb_to_grayscale_image(img_acq)
	diff_w, diff_h = relative_diff(gray)
	diff_2 = torch.pow(diff_w, 2) + torch.pow(diff_h, 2)
	quant = torch.quantile(diff_2, 0.99)
	max_v = diff_2.max()
	print(f'quant = {quant}, diff_2 max = {max_v}')
	removed_outlier = torch.where(diff_2 <= quant, diff_2, quant)

	# abs_w = torch.log(torch.abs(diff_w) + torch.abs(diff_h) +1)
	(a, b, c, d) = false_colour(removed_outlier)
	save_image(d, 'out/diff_normal_pic.png')
