import torch
import torchvision
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2.functional import resize, rgb_to_grayscale_image
from torchvision.utils import save_image

from src.common.common_utils import AppLog, acquire_image, quincunx_diff_avg
from src.encoder_decoder.image_reconstruction_loss import MultiscalePerceptualLoss


def find_central_diff(img):
	img_diff_w = (img[..., 2:] - img[..., :-2]) / 2
	img_diff_h = (img[..., 2:, :] - img[..., :-2, :]) / 2
	return img_diff_w, img_diff_h


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

def test_weight(size):
	traindevice = "cuda" if torch.cuda.is_available() else "cpu"
	with torch.no_grad():
		image = acquire_image('data/CC/train/image_1000.jpeg')
		# image = acquire_image('data/normal_pic.jpg')
		image = image.unsqueeze(0)
		image = image.to(traindevice)
		image = resize(image, [size], InterpolationMode.BILINEAR, antialias=True)
		loss_fn = MultiscalePerceptualLoss().to(traindevice)
		weight = loss_fn.weighted_pixel_imp(image)
		AppLog.info(f'Weight shape: {weight.shape}, image shape: {image.shape}')
		image_pil = torchvision.transforms.ToPILImage()(image.squeeze(0))
		encoded_pil = torchvision.transforms.ToPILImage()(weight.squeeze(0))
		image_pil.show()
		encoded_pil.show()



if __name__ == "__main__":
	test_weight(512)
	# img_acq = acquire_image('data/normal_pic.jpg')
	# gray = rgb_to_grayscale_image(img_acq)
	# # diff_w, diff_h = relative_diff(gray)
	# diff_w, diff_h, avg_img = quincunx_diff_avg(gray)
	# diff_w_abs = torch.abs(diff_w)
	# diff_h_abs = torch.abs(diff_h)
	# diff_mul = diff_w_abs * diff_h_abs
	# tot = torch.div(-2 * diff_mul + diff_w_abs + diff_h_abs + 2 ** (-18), 1 - diff_mul + 2 ** (-18))
	# tot = torch.div(tot, avg_img * 2)
	# stacked = torch.concat([diff_h_abs, diff_w_abs, diff_h_abs], dim=0)
	# save_image(stacked, 'out/stacked_normal_pic.png')
