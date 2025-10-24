import argparse
import io
import os
from pathlib import Path
from random import Random
from typing import Any

import pandas as pd
import torch
import torchvision
from PIL import Image
from diskcache import Cache
from torch import GradScaler
from torch.nn import SmoothL1Loss
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.io import decode_image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from torchvision.transforms.v2 import ColorJitter, Compose, FiveCrop, Lambda, RandomCrop, RandomHorizontalFlip, \
	RandomVerticalFlip, ToDtype

from src.common.common_utils import AppLog, acquire_image
from src.encoder_decoder.image_reconstruction_loss import SaturationLoss
from src.image_encoder_decoder.image_codec import ImageCodec, encode_decode_from_model, prepare_encoder_data, \
	scale_decoder_data


class ImageFolderDataset(Dataset):
	def __init__(self, path: Path, transform=None, cache_path=None):
		super().__init__()
		self.path = path
		self.files = [picfile for picfile in path.iterdir() if picfile.is_file()]
		self.cache = Cache('cache/') if cache_path is None else Cache(cache_path)
		self.transform = transform

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		if idx in self.cache:
			return self.transform(self.cache.get(idx))
		image_path = self.files[idx]
		decoded = decode_image(str(image_path), mode='RGB')
		self.cache[idx] = decoded
		return self.transform(decoded)


def get_data(batch_size=16, minsize=320):
	transform_train = Compose([ToDtype(dtype=torch.float32, scale=True), RandomCrop(minsize),
	                           ColorJitter(saturation=0.5, brightness=0.5, contrast=0.5)])

	transform_validate = Compose([ToDtype(dtype=torch.float32, scale=True), FiveCrop(minsize)])

	train_set = ImageFolderDataset(Path('data/CC/train'), transform=transform_train, cache_path='cache/CC/train')
	validate_set = ImageFolderDataset(Path('data/CC/validate'), transform=transform_validate,
	                                  cache_path='cache/CC/validate')

	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4,
	                          persistent_workers=True, prefetch_factor=4)
	val_loader = DataLoader(validate_set, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4,
	                        persistent_workers=True, prefetch_factor=4)
	return train_loader, val_loader


class TrainEncoderAndDecoder:
	def __init__(self, model, optimizer, train_device, cycle_sch, save_training_fn, starting_epoch, ending_epoch,
	             vloss=float('inf'), scaler=None):
		loss_fn = [SmoothL1Loss(beta=0.5), SaturationLoss()]
		self.random = Random()
		self.device = train_device
		self.model_orig = model
		self.model = self.model_orig.to(self.device)
		self.save_training_fn = save_training_fn
		self.optimizer = optimizer

		self.current_epoch = starting_epoch
		self.ending_epoch = ending_epoch
		self.best_vloss = vloss

		self.trained_one_batch = False

		self.loss_func = loss_fn
		self.scheduler = cycle_sch
		self.scaler = GradScaler() if scaler is None else scaler
		self.train_transform = Compose([RandomVerticalFlip(0.5), RandomHorizontalFlip(0.5)])
		self.validate_transform = Compose([Lambda(lambda crops: tv_tensors.wrap(crops, like=crops[0]))])

	def get_loss_by_inference(self, data, ratio):
		prep = prepare_encoder_data(data)
		with torch.autocast(device_type=self.device):
			encoded, h, w = self.model(prep)
			decoded = self.model(encoded, h, w)
			result = scale_decoder_data(decoded)

			c = encoded.shape[-3]
			decode_channel_ratio = round(c * ratio)
			partial_latent_decode_mask = torch.zeros_like(encoded, device=self.device)
			partial_latent_decode_mask[:, :decode_channel_ratio, :, :] = 1

			partial_latent_decode = encoded * partial_latent_decode_mask
			partial_latent_rest = (1 - partial_latent_decode_mask) * encoded

			partial_decoded = self.model(partial_latent_decode, h, w)
			rest_decoded = decoded - partial_decoded
			encoded_latent, _, _ = self.model(rest_decoded)

			smooth_loss = self.loss_func[0](result, data)
			sat_loss = self.loss_func[1](result, data)
			round_trip_loss = self.loss_func[0](encoded_latent, partial_latent_rest)

		return smooth_loss, sat_loss, round_trip_loss

	def get_loss_validation(self, data, ratio):
		prep = prepare_encoder_data(data)
		with torch.autocast(device_type=self.device):
			encoded, h, w = self.model(prep)
			c = encoded.shape[-3]
			decode_channel_ratio = round(c * ratio)
			partial_latent_decode_mask = torch.zeros_like(encoded, device=self.device)
			partial_latent_decode_mask[:, :decode_channel_ratio, :, :] = 1
			partial_latent_decode = encoded * partial_latent_decode_mask
			partial_decoded = self.model(partial_latent_decode, h, w)
			result = scale_decoder_data(partial_decoded)

			smooth_loss = self.loss_func[0](result, data)
			sat_loss = self.loss_func[1](result, data)

		return smooth_loss + sat_loss

	@torch.compile(mode='max-autotune')
	def train_compilable(self, data: torch.Tensor, ratio: float) -> tuple[Any, Any, Any, Any]:
		data = self.train_transform(data)
		smooth_loss, sat_loss, round_trip_loss = self.get_loss_by_inference(data, ratio)
		return data, smooth_loss, sat_loss, round_trip_loss

	@torch.compile(mode='max-autotune-no-cudagraphs')
	def validate_compiled(self, stacked: torch.Tensor) -> tuple[
		torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		s, n, c, h, w = stacked.shape
		reshaped = torch.reshape(stacked, (s * n, c, h, w))
		smooth_loss1, sat_loss1, round_trip_loss1 = self.get_loss_by_inference(reshaped, self.ratio_val)
		smooth_loss2 = self.get_loss_validation(reshaped, self.ratio_val)
		return smooth_loss1, sat_loss1, round_trip_loss1, smooth_loss2, reshaped

	def train_one_epoch(self, train_loader):
		self.model.train(True)

		pics_seen = 0

		t_loss = {
			'smooth_loss': 0.0,
			'sat_loss': 0.0,
			'round_trip_loss': 0.0
		}

		for batch_idx, data in enumerate(train_loader):
			ratio = self.random.random() * 0.9 + 0.05
			self.optimizer.zero_grad(set_to_none=True)
			data = data.to(self.device)

			data, smooth_loss, sat_loss, round_trip_loss = self.train_compilable(data, ratio)
			loss = self.multiplier * (smooth_loss + sat_loss) + round_trip_loss
			scaled_loss = self.scaler.scale(loss)

			pics_seen += data.shape[0]
			scaled_loss.backward()
			self.scaler.step(self.optimizer)
			self.scaler.update()
			t_loss['smooth_loss'] += smooth_loss.item()
			t_loss['sat_loss'] += sat_loss.item()
			t_loss['round_trip_loss'] += round_trip_loss.item()
			self.scheduler.step()
			if not self.trained_one_batch:
				self.trained_one_batch = True
				AppLog.info(f'Training loss: {t_loss}, batch: {batch_idx + 1}')
		batches = len(train_loader)
		t_loss['smooth_loss'] /= batches
		t_loss['sat_loss'] /= batches
		t_loss['round_trip_loss'] /= batches
		return t_loss, pics_seen

	def evaluate(self, val_loader):
		self.model.eval()
		vloss = {
			'smooth_loss': 0.0,
			'sat_loss': 0.0,
			'round_trip_loss': 0.0,
			'smooth_loss_10p': 0.0
		}
		pics_seen = 0
		with torch.no_grad():
			for batch_idx, data, in enumerate(val_loader):
				transformed = self.validate_transform(data)
				stacked = torch.stack(transformed)
				stacked = stacked.to(self.device)
				smooth_loss1, sat_loss1, round_trip_loss1, smooth_loss2, reshaped = self.validate_compiled(stacked)
				vloss['smooth_loss'] += smooth_loss1.item()
				vloss['sat_loss'] += sat_loss1.item()
				vloss['round_trip_loss'] += round_trip_loss1.item()
				vloss['smooth_loss_10p'] += smooth_loss2.item()

				if batch_idx == 0:
					AppLog.info(f'Stacked shape: {stacked.shape}, reshaped shape: {reshaped.shape}, vloss : {vloss}')
				pics_seen += reshaped.shape[0]
		batches = len(val_loader)
		vloss['smooth_loss'] /= batches
		vloss['sat_loss'] /= batches
		vloss['round_trip_loss'] /= batches
		vloss['smooth_loss_10p'] /= batches

		return vloss, pics_seen

	def train_and_evaluate(self, train_loader, val_loader):

		AppLog.info(f'Training from {self.current_epoch} to {self.ending_epoch} epochs.')
		while self.current_epoch < self.ending_epoch:
			train_loss, p_t = self.train_one_epoch(train_loader)
			val_loss_dic, p_v = self.evaluate(val_loader)
			AppLog.info(
				f'Epoch {self.current_epoch + 1}: Training loss = {train_loss} ({p_t} samples), Validation Loss = '
				f'{val_loss_dic}  ({p_v} samples),  '
				f'lr = {(self.scheduler.get_last_lr()[0]):.3e} ')
			val_loss = val_loss_dic['smooth_loss'] + val_loss_dic['sat_loss'] + val_loss_dic['round_trip_loss']
			if val_loss < self.best_vloss:
				self.best_vloss = val_loss
				self.save_training_fn(self.model_orig, self.optimizer, self.current_epoch + 1, val_loss,
				                      self.scheduler,
				                      self.scaler)
			self.current_epoch += 1


def prepare_data():
	os.makedirs('data/CC/train', exist_ok=True)
	os.makedirs('data/CC/validate', exist_ok=True)

	# Read and concatenate parquet files
	df1 = pd.read_parquet('data/train-00000-of-00002.parquet')
	df2 = pd.read_parquet('data/train-00001-of-00002.parquet')
	combined_df = pd.concat([df1, df2], ignore_index=True)
	AppLog.info(f'Combined dataset size: {len(combined_df)}')
	combined_df = combined_df.sample(frac=1).reset_index(drop=True)
	total = len(combined_df)
	AppLog.info(f'Dataset shuffled: {total}')
	for i, row in enumerate(combined_df.iterrows()):
		image_data = row[1]['image']
		image_bytes = image_data['bytes']
		try:
			image_bytes_raw = io.BytesIO(image_bytes)
			image = Image.open(image_bytes_raw)
			file_name = f'data/CC/train/image_{i}.jpeg' if i < total * 0.8 else f'data/CC/validate/image_{i}.jpeg'
			image.save(file_name)
		except Exception as e:
			AppLog.error(f'Error reading image {i}: {e}')
		if i % 100 == 0:
			AppLog.info(f'Images saved: {i}')


def save_training_state(location, model, optimizer, epoch, vloss, scheduler=None, scaler=None):
	scheduler_state = None if scheduler is None else scheduler.state_dict()
	scaler_state = None if scaler is None else scaler.state_dict()

	torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch,
	            'scheduler_state_dict': scheduler_state, 'v_loss': vloss, 'scaler_state_dict': scaler_state},
	           location, )


def load_training_state(location, model, optimizer=None, scheduler=None, scaler=None):
	checkpoint = torch.load(location)
	model.load_state_dict(checkpoint['model_state_dict'])
	if optimizer is not None and 'optimizer_state_dict' in checkpoint.keys() and checkpoint[
		'optimizer_state_dict'] is not None:
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	if scheduler is not None and 'scheduler_state_dict' in checkpoint.keys() and checkpoint[
		'scheduler_state_dict'] is not None:
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
	if scaler is not None and 'scaler_state_dict' in checkpoint.keys() and checkpoint['scaler_state_dict'] is not None:
		scaler.load_state_dict(checkpoint['scaler_state_dict'])

	epoch = checkpoint['epoch']
	vloss = checkpoint['v_loss']
	return model, optimizer, epoch, vloss, scheduler, scaler


def train_codec(lr_min_arg, lr_max_arg, batch_size, size, reset_vloss, start_new):
	save_location = 'checkpoints/encode_decode/train_codec.pth'
	traindevice = "cuda" if torch.cuda.is_available() else "cpu"
	enc = getImageEncoderDecoder().to(traindevice)
	lr_min = lr_min_arg if lr_min_arg > 0 else 1e-3
	lr_max = lr_max_arg if lr_max_arg > 0 else 1e-2
	decay_params = []
	no_decay_params = []

	for name, param in enc.named_parameters():
		if not param.requires_grad:
			continue
		# Apply weight decay only to convolutional layers, not to normalization layers
		if 'GroupNorm' in name or 'norm' in name or 'bn' in name or 'bias' in name:
			no_decay_params.append(param)
		else:
			decay_params.append(param)

	optimizer = torch.optim.NAdam([{'params': decay_params, 'weight_decay': 1e-4},{'params': no_decay_params, 'weight_decay': 0}], lr=lr_min, decoupled_weight_decay=True)
	max_epochs = 30
	train_loader, val_loader = get_data(batch_size=batch_size, minsize=size)
	save_training_fn = lambda enc_p, optimizer_p, epoch_p, vloss_p, sch, sc: save_training_state(save_location, enc_p,
	                                                                                             optimizer_p, epoch_p,
	                                                                                             vloss_p, sch, sc)
	# linearLr = LinearLR(optimizer, start_factor=lr_max, total_iters=8)
	# cosinelr = CosineAnnealingLR(optimizer, T_max=(max_epochs - 9))
	# # cyc_sch = SequentialLR(optimizer, schedulers=[linearLr, cosinelr], milestones=[9])
	cyc_sch = OneCycleLR(optimizer, max_lr=lr_max, epochs=max_epochs, steps_per_epoch=len(train_loader))

	if os.path.exists(save_location) and not start_new:
		if lr_max_arg > 0 and lr_min_arg > 0:
			enc, optim, epoch, vloss, scheduler, scaler = load_training_state(save_location, enc, None, None, None)
			scheduler = cyc_sch
			optim = optimizer
		else:
			enc, optim, epoch, vloss, scheduler, scaler = load_training_state(save_location, enc, optimizer, cyc_sch,
			                                                                  GradScaler())
		AppLog.info(f'Loaded checkpoint from epoch {epoch} with vloss {vloss:.3e} and scheduler {scheduler}')
		if reset_vloss:
			vloss = float('inf')
			epoch = 0
		AppLog.info(f'(Re)Starting from epoch {epoch} with vloss {vloss:.3e} and scheduler {scheduler}, using device '
		            f'{traindevice}')

		trainer = TrainEncoderAndDecoder(enc, optim, traindevice, scheduler, save_training_fn, epoch, max_epochs,
		                                 vloss,
		                                 scaler)
		trainer.train_and_evaluate(train_loader, val_loader)
	else:
		AppLog.info(
			f'Training from scratch. Using lr_min={lr_min}, lr_max={lr_max} and scheduler {cyc_sch}, using device '
			f'{traindevice}')
		trainer = TrainEncoderAndDecoder(enc, optimizer, traindevice, cyc_sch, save_training_fn, 0, max_epochs)
		trainer.train_and_evaluate(train_loader, val_loader)


def getImageEncoderDecoder():
	# codec = get_simple_encoder_decoder()
	codec = ImageCodec(64, 512, 48)
	return codec


def test_and_show(size):
	save_location = 'checkpoints/encode_decode/train_codec.pth'
	enc = getImageEncoderDecoder()
	traindevice = "cuda" if torch.cuda.is_available() else "cpu"
	if os.path.exists(save_location):
		enc, optimizer, epoch, vloss, _, _ = load_training_state(save_location, enc)
		AppLog.info(f'Loaded checkpoint from epoch {epoch} with vloss {vloss:.3e}')
		enc.eval()
		enc.to(traindevice)
		with torch.no_grad():
			# image = acquire_image('data/CC/train/image_1000.jpeg')
			image = acquire_image('data/reddit_face.jpg')
			image = image.unsqueeze(0)
			image = image.to(traindevice)
			image = resize(image, [size], InterpolationMode.BILINEAR, antialias=True)
			print(image.shape)
			encoded, lat = encode_decode_from_model(enc, image)
			encoded = torch.clamp(encoded, 0, 1)
			image_pil = torchvision.transforms.ToPILImage()(image.squeeze(0))
			encoded_pil = torchvision.transforms.ToPILImage()(encoded.squeeze(0))
			image_pil.show()
			encoded_pil.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train encoder and decoder model')
	parser.add_argument('--lr-min', type=float, default=-1, help='Min learning rate for training')
	parser.add_argument('--lr-max', type=float, default=-1, help='Max learning rate for training')
	parser.add_argument('--batch-size', type=int, default=12, help='Batch size for training')
	parser.add_argument('--size', type=int, default=300, help='Image size for training and validation')
	parser.add_argument('--start-new', type=bool, default=False, help='Start new training instead of resuming')
	parser.add_argument('--reset-vloss', type=bool, default=False, help='Prepare data for training')
	parser.add_argument('--test', type=bool, default=False, help='Show a reconstructed test image')
	args = parser.parse_args()
	if args.test:
		test_and_show(size=args.size)
	else:
		train_codec(args.lr_min, args.lr_max, args.batch_size, args.size, args.reset_vloss, args.start_new)
	AppLog.shut_down()
