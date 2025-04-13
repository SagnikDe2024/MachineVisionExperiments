import itertools
import pickle
from pathlib import Path

import datasets
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2.functional import crop_image, normalize, rotate, to_dtype, to_pil_image

from src.utils.common_utils import AppLog


class RawImageDataSet(Dataset):
	def __init__(self, directory: Path):
		super().__init__()
		self.directory = directory
		image_files = itertools.chain(directory.glob("*.jpg"), directory.glob("*.jpeg"), directory.glob("*.png"))
		self.image_files = list(image_files)

	def __len__(self):
		return len(self.image_files)

	def __getitem__(self, idx):
		image_path = self.image_files[idx]
		float_tensor_image = to_dtype(decode_image(image_path, mode='RGB'), dtype=torch.float32, scale=True)
		# Generate an id for the image
		image_id = hash(pickle.dumps(float_tensor_image))
		return image_id, float_tensor_image


class Dice(torch.nn.Module):
	def __init__(self, size, seed=0):
		super().__init__()
		self.size = size
		self.seed = seed
		# self.rand_cuda = torch.Generator(device='cuda')
		self.rand_cuda = torch.Generator()
		if seed != 0:
			self.rand_cuda.manual_seed(seed)

	def forward(self, pic, random_slice=False):
		pic = torch.squeeze(pic)
		(c, h, w) = pic.shape

		if random_slice:
			top_lefts = self.random_cropping(h, w)
		else:
			h_slices = h // self.size
			w_slices = w // self.size
			rest_h = h - h_slices * self.size
			rest_w = w - w_slices * self.size
			last_h = torch.tensor([h - self.size], dtype=torch.int32) if rest_h > 0 else torch.empty(0,
																									 dtype=torch.int32)
			last_w = torch.tensor([w - self.size], dtype=torch.int32) if rest_w > 0 else torch.empty(0,
																									 dtype=torch.int32)
			h_offsets = torch.arange(0, h_slices, dtype=torch.int32) * self.size
			w_offsets = torch.arange(0, w_slices, dtype=torch.int32) * self.size
			all_h = torch.concat([h_offsets, last_h])
			all_w = torch.concat([w_offsets, last_w])
			top_lefts = torch.cartesian_prod(all_h, all_w)

		diff_h, diff_w, prepared_image = normalize_and_getdiffs(pic)
		diff_h_min = diff_h.amin(dim=[1, 2])

		# std here basically means range. std is used as that's what pytorch normalize says

		diff_h_std = diff_h.amax(dim=[1, 2]) - diff_h_min
		diff_w_min = diff_w.amin(dim=[1, 2])
		diff_w_std = diff_w.amax(dim=[1, 2]) - diff_w_min

		diff_h_min_tolist = diff_h_min.tolist()
		diff_h_std_tolist = diff_h_std.tolist()
		diff_w_min_tolist = diff_w_min.tolist()
		diff_w_std_tolist = diff_w_std.tolist()

		diff_h_norm = normalize(diff_h, diff_h_min_tolist, diff_h_std_tolist)

		diff_w_norm = normalize(diff_w, diff_w_min_tolist, diff_w_std_tolist)

		data = {'diff_h_min_rgb': diff_h_min_tolist, 'diff_h_std_rgb': diff_h_std_tolist,
				'diff_w_min_rgb': diff_w_min_tolist, 'diff_w_std_rgb': diff_w_std_tolist, 'height': h, 'width': w}

		def get_slices(top_left):
			t = top_left[0]
			l = top_left[1]
			cropped_image = crop_image(pic, t, l, self.size, self.size)
			cropped_diff_h = crop_image(diff_h_norm, t, l, self.size, self.size)
			cropped_diff_w = crop_image(diff_w_norm, t, l, self.size, self.size)
			return t, l, cropped_image, cropped_diff_h, cropped_diff_w

		slices = list(map(get_slices, top_lefts))
		data['slices'] = slices

		return data

	def random_cropping(self, h: int, w: int):
		times = int((h // self.size) * (w // self.size) * 4 / 3)
		AppLog.info(f'Random cropping with times: {times}')
		rand_slices = torch.rand((2, times), generator=self.rand_cuda, requires_grad=False)
		top_and_lefts = torch.concat([rand_slices[0] * (h - self.size), rand_slices[1] * (w - self.size)]).to(
				dtype=torch.int32)
		top_lefts = torch.permute(top_and_lefts, (1, 0))
		return top_lefts


def normalize_and_getdiffs(float_tensor_image):

	prepared_image = normalize(float_tensor_image, 0.5, 0.5)

	img_45 = rotate(prepared_image, 45, interpolation=InterpolationMode.BILINEAR, expand=True)
	(_, h45, w45) = img_45.shape
	(_, h, w) = prepared_image.shape
	top, left = (h45 - h) // 2, (w45 - w) // 2
	diff_45_w = torch.diff(img_45, dim=-1)
	diff_45_h = torch.diff(img_45, dim=-2)

	diff_rotback_w = rotate(diff_45_w, -45, interpolation=InterpolationMode.BILINEAR, expand=False)
	diff_rotback_h = rotate(diff_45_h, -45, interpolation=InterpolationMode.BILINEAR, expand=False)
	diff_rotback_w_cropped = crop_image(diff_rotback_w, top, left, h, w)

	diff_rotback_h_cropped = crop_image(diff_rotback_h, top, left, h, w)

	diff_w = diff_rotback_w_cropped - diff_rotback_h_cropped
	diff_h = diff_rotback_w_cropped + diff_rotback_h_cropped

	return diff_h, diff_w, prepared_image


class PrepareData:
	def __init__(self, in_location: Path, out_location: Path):
		self.in_location = in_location
		self.out_location = out_location
		self.dataset = RawImageDataSet(in_location)
		self.dicing = torch.compile(Dice(128))

	# self.dicing = torch.compile(Dice(128), mode="max-autotune")

	def prepare_images(self):
		AppLog.info(f'There are {len(self.dataset)} images')
		self.dicing.cuda()

		data_loader = DataLoader(self.dataset)

		for data in data_loader:
			img_id, tensor_image = data
			(n, c, h, w) = tensor_image.shape
			tensor_image = tensor_image.cuda()
			data = self.dicing.forward(tensor_image)
			# AppLog.info(f'Diced images {diced_images}')
			diced_images = data['slices']
			tops, lefts, imgs, img_hs, img_ws = zip(*diced_images)

			# diced_images = map(lambda y: (y[0].cpu(), y[1].cpu(), y[2].cpu(), y[3].cpu(), y[4].cpu()), diced_images)
			image_slices = {'top'  : list(map(lambda t: t.item(), tops)),
							'left' : list(map(lambda l: l.item(), lefts)),
							'img'  : list(map(lambda x: to_pil_image(x, mode='RGB'), imgs)),
							'img_h': list(map(lambda x: to_pil_image(x, mode='RGB'), img_hs)),
							'img_w': list(map(lambda x: to_pil_image(x, mode='RGB'), img_ws))}

			dataframe = pd.DataFrame(image_slices)
			rows = dataframe.shape[0]
			dataframe['diff_h_min_rgb'] = [data['diff_h_min_rgb'] for _ in range(rows)]
			dataframe['diff_h_std_rgb'] = [data['diff_h_std_rgb'] for _ in range(rows)]
			dataframe['diff_w_min_rgb'] = [data['diff_w_min_rgb'] for _ in range(rows)]
			dataframe['diff_w_std_rgb'] = [data['diff_w_std_rgb'] for _ in range(rows)]

			dataframe['width'] = w
			dataframe['height'] = h

			dataframe['image_id'] = img_id.item()

			AppLog.info(f'dataframe shape = {dataframe.shape}')
			dictionary = dataframe.to_dict(orient='list')
			dataset = datasets.Dataset.from_dict(dictionary)
			dataset.save_to_disk(self.out_location / f'img_id_{img_id.item()}.parquet')

		# dataframe.to_parquet(self.out_location / f'img_id_{img_id.item()}.parquet', index=True)
