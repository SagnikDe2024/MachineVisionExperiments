import torch
from torchvision.transforms.v2.functional import rgb_to_grayscale_image
from torchvision.utils import save_image

from src.common.common_utils import acquire_image


def find_central_diff(img):
	img_diff_w = (img[..., 2:] - img[..., :-2]) / 2
	img_diff_h = (img[..., 2:, :] - img[..., :-2, :]) / 2
	return img_diff_w, img_diff_h


def quincunx_diff_avg(img):
	top_left = img[..., :-1, :-1]
	top_right = img[..., :-1, 1:]
	bottom_left = img[..., 1:, :-1]
	bottom_right = img[..., 1:, 1:]

	img_diff_w = ((top_right - bottom_left) + (bottom_right - top_left)) / 2
	img_diff_h = (-(top_right - bottom_left) + (bottom_right - top_left)) / 2
	img_avg = (top_right + bottom_left + bottom_right + top_left) / 4
	return img_diff_w, img_diff_h, img_avg


def relative_diff(img):
	diff_w, diff_h, avg = quincunx_diff_avg(img)
	rel_diff_w = diff_w / (avg + 2 ** (-18))
	rel_diff_h = diff_h / (avg + 2 ** (-18))
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
	# diff_w, diff_h = relative_diff(gray)
	diff_w, diff_h, avg_img = quincunx_diff_avg(gray)
	diff_w_abs = torch.abs(diff_w)
	diff_h_abs = torch.abs(diff_h)
	diff_mul = diff_w_abs * diff_h_abs
	tot = torch.div(-2 * diff_mul + diff_w_abs + diff_h_abs + 2 ** (-18), 1 - diff_mul + 2 ** (-18))
	tot = torch.div(tot, avg_img * 2)
	stacked = torch.concat([diff_h_abs, diff_w_abs, diff_h_abs], dim=0)
	save_image(stacked, 'out/stacked_normal_pic.png')
