from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import fractional_max_pool2d
from torchvision import transforms
from torchvision.io import ImageReadMode, decode_image, write_jpeg
from torchvision.transforms.v2.functional import InterpolationMode, crop_image, normalize, resize_image, rotate

from src.encoder_decoder.prepare_data import PrepareData, normalize_and_getdiffs
from src.common.common_utils import AppLog

below = 0.01


def conv_test(image_file: str) -> None:
	img = decode_image(image_file, mode=ImageReadMode.RGB)

	# transform = v2.Compose([v2.Resize((512, 512))])
	# resized = transform(img)
	perform_fft_and_save(img)


# perform_fft_and_save(resized)


def filter_before_writing(img):
	img_f = img.to(dtype=torch.float)
	img_f_mi = img_f.min()
	img_f_ma = img_f.max()
	if img_f_mi < 0 or img_f_ma > 255:
		nine_five = torch.quantile(img_f, 1 - below)
		five = torch.quantile(img_f, below)
		clamped = torch.clamp(img_f, five, nine_five)
		img_f_scaled = (clamped - five) * 255 / (nine_five - five)
		print(f'Filtered -> {below} -> {five} , {1 - below} -> {nine_five} ')
		return img_f_scaled
	else:
		return img_f


def perform_fft_and_save(img: torch.Tensor) -> None:
	summed = img.sum()
	print(f'size = {img.shape} and sum = {summed}')

	pic_fft = torch.fft.fft2(img)
	pic_fft_abs = pic_fft.abs()
	pic_fft_angle = pic_fft.angle()
	trim_below = torch.quantile(pic_fft_abs, below)
	trim_above = torch.quantile(pic_fft_abs, 1 - below)
	clamped_abs = torch.clamp(pic_fft_abs, trim_below, trim_above)
	pic_fft_clamped = torch.multiply(clamped_abs, torch.exp(1j * pic_fft_angle))

	print(f'min = {pic_fft_abs.min()} , median -> {pic_fft_abs.median()} , max -> {pic_fft_abs.max()} ')
	print(f'Clamped {below} = {trim_below} , {1 - below} -> {trim_above}')

	# clamped_log = torch.log(clamped_abs)
	hist = torch.histc(pic_fft_abs, 256, min=trim_below, max=trim_above)
	bins = np.arange(0, 256)
	plt.plot(bins, hist)
	plt.show()

	fft_real = pic_fft_clamped.real
	fft_imag = pic_fft_clamped.imag

	converted = filter_before_writing(torch.fft.ifft2(pic_fft_clamped))
	converted_re = filter_before_writing(torch.fft.ifft2(fft_real))
	converted_im = filter_before_writing(torch.abs(torch.fft.ifft2(fft_imag).imag))
	write_jpeg(converted.to(dtype=torch.uint8), "../out/converted.jpg")
	write_jpeg(converted_re.to(dtype=torch.uint8), "../out/converted_re.jpg")
	write_jpeg(converted_im.to(dtype=torch.uint8), "../out/converted_im.jpg")


def gen_fourier_image(size, angle):
	radius = size // 2
	max_radius = radius * (2 ** 0.5)
	diameter = radius * 2
	max_value = diameter ** 2 * 127
	p = np.log(max_value) / np.log(max_radius)
	# pic2d = np.zeros((diameter, diameter), dtype=np.complex64)
	phases = np.random.rand(diameter, diameter)
	pic2d = np.sin(phases * 2 * np.pi) + 1j * np.cos(phases * 2 * np.pi)
	for y in range(diameter):
		for x in range(diameter):
			r2 = ((x - radius) ** 2 + (y - radius) ** 2)
			if r2 > 0:
				abs_val = max_value * (r2 ** (-p / 2))
				# f_component = abs_val*phase
				pic2d[y, x] *= abs_val
	image_with_fft = torch.tensor(pic2d).unsqueeze(0)
	conv = torch.fft.fft2(image_with_fft)
	print(f'Shape = {conv.shape}')

	# convreal = conv.real
	convreal = torch.abs(conv)
	max_value_r = convreal.max()
	min_value_r = convreal.min()
	med = torch.median(convreal).item()
	# Maximum value and median assuming min_value_r is zero
	max_v = max_value_r - min_value_r
	med_v = med - min_value_r
	# Assume that pixel value p with range [0,255]
	# now A 255^n = max_v and A 128^n = med_v
	# then we have n = log2(max_v/med_v) and A = max_v/(255^n)
	n = np.log2(max_v / med_v)
	A = max_v / (255 ** n)
	# So now p = (y/A)^(1/n) for pixel value
	normed = torch.float_power(torch.divide(torch.subtract(convreal, min_value_r), A), 1 / n)

	print(f'Max = {max_value_r}, Min = {min_value_r} and median = {med}')
	write_jpeg(normed.to(dtype=torch.uint8), "../out/ifft_n.jpg")
	return normed


def minimize_and_rotate(t1, times):
	angle = 90 / times
	rot_image = []
	for _ in range(times):
		c, h, w = t1.shape
		h, w = h // 2, w // 2
		t1 = resize_image(t1, [h, w])
		t1 = rotate(t1, angle, interpolation=InterpolationMode.BILINEAR, expand=True)
		rot_image.append(t1)
	return rot_image


def find_multiscale_diff(rot_images: List[torch.Tensor], times, crop, scale=1.5, strength=0.5):

	top, left, imgh, imgw = crop
	print(f'Cropping {crop}')
	zero_img = torch.zeros([1, imgh, imgw]).cuda()
	for t in range(times):

		for i, rot_image in enumerate(rot_images):
			(c, h, w) = rot_image.shape
			# scale_used = (1/scale)**t
			(hr, hw) = round((1 / scale) ** t * h), round((1 / scale) ** t * w)
			print(f'Scaled values, {hr, hw} from {h, w}')
			rot_image_resized = resize_image(rot_image, [hr, hw])
			img_diff_rot = torch.diff(rot_image_resized)
			orig_size_diff_rotated = resize_image(img_diff_rot, [h, w])
			back_rot = -(45 + i * 90)
			print(f'Back rotated image with angle {back_rot}')
			rot_back = rotate(orig_size_diff_rotated, back_rot, interpolation=InterpolationMode.BILINEAR, expand=False)
			orig_size = crop_image(rot_back, top, left, imgh, imgw)
			diff_strength = strength ** t
			zero_img = zero_img + diff_strength * torch.abs(orig_size)

	return zero_img


def use_frac_pool(t1, kernel, fraction, times: int):

	pooled_image = []

	for _ in range(times):
		t1 = fractional_max_pool2d(t1, kernel, output_ratio=fraction)
		pooled_image.append(t1)
	return pooled_image


def just_dosomething():
	global h, w
	transform = transforms.Compose([transforms.Normalize(127.5, 127.5)])
	img_raw = decode_image("data/reddit_face.jpg", mode=ImageReadMode.GRAY).to(dtype=torch.float32).cuda()
	img = transform(img_raw)
	img_45 = rotate(img, 45, interpolation=InterpolationMode.BILINEAR, expand=True)
	img_135 = rotate(img, 135, interpolation=InterpolationMode.BILINEAR, expand=True)
	img_225 = rotate(img, 225, interpolation=InterpolationMode.BILINEAR, expand=True)
	img_315 = rotate(img, 315, interpolation=InterpolationMode.BILINEAR, expand=True)
	(c, h, w) = img_45.shape
	(c2, h2, w2) = img.shape
	top, left = (h - h2) // 2, (w - w2) // 2
	diff = find_multiscale_diff([img_45, img_135, img_225, img_315], 8, (top, left, h2, w2), 1.5, 1.5)
	# img_45_diff = rotate(torch.diff(img_45),-45, interpolation=InterpolationMode.BILINEAR, expand=False)
	# img_135_diff = rotate(torch.diff(img_135),-135, interpolation=InterpolationMode.BILINEAR, expand=False)
	# img_225_diff = rotate(torch.diff(img_225),-225, interpolation=InterpolationMode.BILINEAR, expand=False)
	# img_315_diff = rotate(torch.diff(img_315),-315, interpolation=InterpolationMode.BILINEAR, expand=False)
	# img_diff = torch.abs(img_45_diff) +torch.abs(img_135_diff) + torch.abs(img_225_diff) + torch.abs(img_315_diff)
	# img_diff = crop_image(img_diff, top, left,h2,w2)
	img_diff = diff
	print(f'top = {top}, left = {left}, h = {h2}, w = {w2} , top+h2 = {top + h2}, left+w2 = {left + w2}')
	max_2, max_4, max_value, result = false_colour(img_diff)
	# med_v2 = med_value / 2
	print(f'Max = {max_value}, Max2 = {max_2}, Max4 = {max_4}')
	ones = torch.ones_like(img_diff)
	# result = torch.where(img_diff < med_v2, torch.concat([zero_img, zero_img, img_diff / med_v2], dim=0),
	# 			torch.where(img_diff < med_value, torch.concat([(img_diff - med_v2)/(med_value-med_v2),zero_img,
	# 			ones], dim=0 ),
	# 						torch.concat([ones,(img_diff - med_value)/(max_value - med_value), zero_img], dim=0)))
	result = result * 255
	write_jpeg(result.to(dtype=torch.uint8).cpu(), "out/face_diff.jpg")


def false_colour(img_diff):
	max_value = img_diff.max()
	mean_v = img_diff.mean()

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


def test_prepdata():
	global h, w
	prepareData = PrepareData()
	diff_h, diff_w, h, prepared_image, w = normalize_and_getdiffs('data/reddit_face.jpg')
	tot_diff = torch.pow(torch.pow(diff_h, 2) + torch.pow(diff_w, 2), 0.5)

	diff_h_min = diff_h.amin(dim=[1, 2])
	diff_h_std = diff_h.amax(dim=[1, 2]) - diff_h_min
	diff_w_min = diff_w.amin(dim=[1, 2])
	diff_w_std = diff_w.amax(dim=[1, 2]) - diff_w_min
	imgmin = prepared_image.amin(dim=[1, 2])
	imgstd = prepared_image.amax(dim=[1, 2]) - imgmin
	diff_h_norm = normalize(diff_h, diff_h_min.tolist(), (diff_h_std / 255.0).tolist())
	diff_w_norm = normalize(diff_w, diff_w_min.tolist(), (diff_w_std / 255.0).tolist())
	AppLog.info(f'diffh is {diff_h_min} and std is {diff_h_std}')
	AppLog.info(f'diffw is {diff_w_min} and std is {diff_w_std}')
	AppLog.info(f'imgmin is {imgmin} and std is {imgstd}')
	unnormal_img = normalize(prepared_image, imgmin.tolist(), (imgstd / 255.0).tolist())
	unnormal_diffh = diff_h_norm  ##normalize(diff_w_norm, -1, 1/127.5)
	unnormal_diffw = diff_w_norm  ## normalize(diff_h_norm, -1, 1/127.5)
	write_jpeg(unnormal_img.to(dtype=torch.uint8), "out/unnormal_img.jpg")
	write_jpeg(unnormal_diffw.to(dtype=torch.uint8), "out/unnormal_diffw.jpg")
	write_jpeg(unnormal_diffh.to(dtype=torch.uint8), "out/unnormal_diffh.jpg")
	AppLog.shut_down()


def check_slice_dice():
	this_path = Path.cwd()
	AppLog.info('Current working directory: {}'.format(this_path))
	in_path = this_path / 'data'
	out_path = this_path / 'out'
	torch.cuda.device(0)

	prepareData = PrepareData(in_path, out_path)
	prepareData.prepare_images()
	AppLog.shut_down()


if __name__ == '__main__':
	check_slice_dice()

# test_prepdata()

# just_dosomething()

# frac_pool = use_frac_pool(img, (2, 2), (2 / 3, 2 / 3), 10)
# # rotated_images = minimize_and_rotate(img, 6)
# trans_imgs = frac_pool
# for i, rotated_image in enumerate(trans_imgs):
# 	write_png(rotated_image.to(dtype=torch.uint8), f'../out/frac_pool_{i}_image.png')

# img = img.to(dtype=torch.float32)
# img = img/255
# crop_size = 256
# padding = crop_size//4
# img = crop(img,400,1244 ,  crop_size,crop_size)
# increased_img = pad_image(img,padding=padding)
#
# increased_img_45 = rotate(increased_img,45,InterpolationMode.BILINEAR)
# increased_img_135 = rotate(increased_img,-45,InterpolationMode.BILINEAR)
# rot_x_diff = torch.diff(increased_img_45,dim=2)
# rot_y_diff = torch.diff(increased_img_135,dim=2)
#
# x_diff = rotate(rot_x_diff,-45,InterpolationMode.BILINEAR)
# y_diff = rotate(rot_y_diff,45,InterpolationMode.BILINEAR)
# x_diff_abs = torch.abs(x_diff)
# y_diff_abs = torch.abs(y_diff)
# max_diff = (x_diff_abs + y_diff_abs - 2*(x_diff_abs*y_diff_abs))/(2-x_diff_abs-y_diff_abs)
# cropped_diff = crop(max_diff,padding-1,padding-1, crop_size+2,crop_size+2)*255

# img = rotate(img,45,InterpolationMode.BILINEAR)
# # img = resize(img,[256,256])
# complex_image = convert_rgb_to_complex(img)
# undiff_complex = convert_complex_to_rgb(complex_image).squeeze(0)
# write_png(undiff_complex.to(dtype=torch.uint8), "../out/complex_crop.png")
# diff = find_complex_diff(complex_image)
# diff_abs_max = diff.abs().max()
# diff_scaled = diff*255 / diff_abs_max
# rgb_image = convert_complex_to_rgb(diff_scaled).squeeze(0)
# # rgb_rotated_back = rotate(rgb_image,-45,InterpolationMode.BILINEAR)

# write_png(cropped_diff.to(dtype=torch.uint8), "../out/complex_pipe_max_crop.png")

# pdf = calculate_pdf(resized)
# log_pdf = np.log2(pdf,out = np.zeros_like(pdf, dtype=np.float32), where = (pdf != 0))
# multiplied = np.multiply(-log_pdf, pdf)
# entropy = np.sum(multiplied)
# print(f'Total entropy = {entropy}')
# plt.contourf(log_pdf[1])
# plt.axis('square')
# plt.show()

# conv_test("../data/normal_pic.jpg")
