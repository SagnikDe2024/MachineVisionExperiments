import argparse
import io
import os
from pathlib import Path

import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from torchvision.transforms.v2 import CenterCrop, RandomCrop, RandomHorizontalFlip, RandomResize, RandomVerticalFlip, \
	Resize

from src.common.common_utils import AppLog, acquire_image
from src.encoder_decoder.image_reconstruction_loss import MultiScaleGradientLoss
from src.image_encoder_decoder.image_codec import ImageCodec, encode_decode_from_model, prepare_encoder_data


class ImageFolderDataset(Dataset):
	def __init__(self, path: Path, transform=None):
		super().__init__()
		self.path = path
		self.files = [picfile for picfile in path.iterdir() if picfile.is_file()]
		self.transform = transform

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		image_path = self.files[idx]
		image = acquire_image(image_path)
		return self.transform(image)


def get_data(batch_size=16, minsize=272):
	maxsize = round(minsize * 2.5)
	transform_train = torchvision.transforms.Compose([
			RandomResize(minsize, maxsize),
			RandomCrop(minsize), RandomVerticalFlip(0.5), RandomHorizontalFlip(0.5),
	])
	transform_validate = torchvision.transforms.Compose([
			Resize(minsize, interpolation=InterpolationMode.BILINEAR),
			CenterCrop(minsize),
	])

	train_set = ImageFolderDataset(Path('data/CC/train'), transform=transform_train)
	validate_set = ImageFolderDataset(Path('data/CC/validate'), transform=transform_validate)

	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
	val_loader = DataLoader(validate_set, batch_size=batch_size, shuffle=False, drop_last=True)
	return train_loader, val_loader


class TrainEncoderAndDecoder:
	def __init__(self, model, optimizer, train_device, cycle_sch, save_training_fn, starting_epoch, ending_epoch,
	             vloss=float('inf')):
		self.device = train_device
		self.model_orig = model
		self.model = torch.compile(self.model_orig, mode="max-autotune").to(self.device)
		self.save_training_fn = save_training_fn
		self.optimizer = optimizer

		self.current_epoch = starting_epoch
		self.ending_epoch = ending_epoch
		self.best_vloss = vloss

		self.trained_one_batch = False
		self.loss_func = torch.compile(MultiScaleGradientLoss(self.device), mode="max-autotune").to(self.device)
		# self.loss_func = torch.compile(ReconstructionLossRelative(), mode="default").to(self.device)
		self.scheduler = cycle_sch(self.optimizer)

	# self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=1 / 3, patience=3,
	#                                                 min_lr=1e-8)

	def train_one_epoch(self, train_loader):
		self.model.train(True)
		tloss = 0.0
		pics_seen = 0
		for batch_idx, data in enumerate(train_loader):
			with torch.backends.cudnn.flags(enabled=True):
				data = data.to(self.device)
				self.optimizer.zero_grad()
				loss = self.get_loss_by_inference(data)
				tloss += loss.item()
				pics_seen += data.shape[0]
				loss.backward()
				self.optimizer.step()
				self.scheduler.step()
			if not self.trained_one_batch:
				self.trained_one_batch = True
				AppLog.info(f'Training loss: {tloss}, batch: {batch_idx + 1}')
		return tloss / pics_seen

	def get_loss_by_inference(self, data):
		with torch.autocast(device_type=self.device):
			result, lat = self.result_enc_dec(data)
			std_result = prepare_encoder_data(result)
			std_data = prepare_encoder_data(data)
			loss = self.loss_func(std_result, std_data)
		return loss

	def result_enc_dec(self, data):
		return encode_decode_from_model(self.model, data)

	def evaluate(self, val_loader):
		self.model.eval()
		vloss = 0.0
		pics_seen = 0
		with torch.no_grad():
			for batch_idx, data, in enumerate(val_loader):
				data = data.to(self.device)
				loss = self.get_loss_by_inference(data)
				vloss += loss.item()
				pics_seen += data.shape[0]
		return vloss / pics_seen

	def train_and_evaluate(self, train_loader, val_loader):

		AppLog.info(f'Training from {self.current_epoch} to {self.ending_epoch} epochs.')
		while self.current_epoch < self.ending_epoch:
			train_loss = self.train_one_epoch(train_loader)
			val_loss = self.evaluate(val_loader)
			# self.scheduler.step(val_loss,epoch=epoch)
			AppLog.info(
					f'Epoch {self.current_epoch + 1}: Training loss = {train_loss:.3e}, Validation Loss = '
					f'{val_loss:.3e}, '
					f'lr = {(self.scheduler.get_last_lr()[0]):.3e}')
			if val_loss < self.best_vloss:
				self.best_vloss = val_loss
				self.save_training_fn(self.model_orig, self.optimizer, self.current_epoch + 1, val_loss)
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


def save_training_state(location, model, optimizer, epoch, vloss):
	torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch,
	            'v_loss'          : vloss, }, location, )


def load_training_state(location, model, optimizer, only_model=False):
	checkpoint = torch.load(location)
	model.load_state_dict(checkpoint['model_state_dict'])
	if not only_model:
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	vloss = checkpoint['v_loss']
	return model, optimizer, epoch, vloss


def train_codec(lr_min, lr_max, batch_size, size, start_new):
	save_location = 'checkpoints/encode_decode/train_codec_L1_loss.pth'
	traindevice = "cuda" if torch.cuda.is_available() else "cpu"
	enc = getImageEncoderDecoder().to(traindevice)
	# get_optim = lambda model :  torch.optim.AdamW(
	# 		[{'params': model.encoder.parameters()},
	# 		 {'params': model.decoder.parameters(), 'weight_decay': 0.0001}],
	# 		lr=lr_min)
	optimizer = torch.optim.AdamW(
			[{'params': enc.encoder.parameters()},
			 {'params': enc.decoder.parameters(), 'weight_decay': 0.0001}],
			lr=lr_min)

	train_loader, val_loader = get_data(batch_size=batch_size, minsize=size)
	save_training_fn = lambda enc_p, optimizer_p, epoch_p, vloss_p: save_training_state(save_location, enc_p,
	                                                                                    optimizer_p, epoch_p, vloss_p)

	if os.path.exists(save_location) and not start_new:
		enc, optimizer, epoch, vloss = load_training_state(save_location, enc, optimizer)
		AppLog.info(f'Loaded checkpoint from epoch {epoch} with vloss {vloss:.3e}')
		scheduler_fn = lambda optim: CyclicLR(optim, base_lr=lr_min, max_lr=lr_max, mode='triangular2')
		trainer = TrainEncoderAndDecoder(enc, optimizer, traindevice, scheduler_fn, save_training_fn, epoch, 30, vloss)
		trainer.train_and_evaluate(train_loader, val_loader)
	else:
		AppLog.info(f'Training from scratch. Using learning rate {lr_min} and device {traindevice}')
		scheduler_fn = lambda optim: CyclicLR(optim, base_lr=lr_min, max_lr=lr_max, mode='triangular2')
		trainer = TrainEncoderAndDecoder(enc, optimizer, traindevice, scheduler_fn, save_training_fn, 0, 30)
		trainer.train_and_evaluate(train_loader, val_loader)


def getImageEncoderDecoder():
	enc = ImageCodec(64, 256, 64, enc_layers=6, dec_layers=6)
	return enc


def test_and_show():
	save_location = 'checkpoints/encode_decode/train_codec_L1_loss.pth'
	enc = getImageEncoderDecoder()
	optimizer = torch.optim.SGD(enc.parameters(), lr=0.1)
	traindevice = "cuda" if torch.cuda.is_available() else "cpu"
	if os.path.exists(save_location):
		enc, optimizer, epoch, vloss = load_training_state(save_location, enc, optimizer, only_model=True)
		AppLog.info(f'Loaded checkpoint from epoch {epoch} with vloss {vloss:.3e}')
		enc.eval()
		enc.to(traindevice)
		with torch.no_grad():
			image = acquire_image('data/CC/train/image_1000.jpeg')
			# image = acquire_image('data/normal_pic.jpg')
			image = image.unsqueeze(0)
			image = image.to(traindevice)
			image = resize(image, [size], InterpolationMode.BILINEAR, antialias=True)
			encoded, lat = encode_decode_from_model(enc, image)
			image_pil = torchvision.transforms.ToPILImage()(image.squeeze(0))
			encoded_pil = torchvision.transforms.ToPILImage()(encoded.squeeze(0))
			image_pil.show()
			encoded_pil.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train encoder and decoder model')
	parser.add_argument('--lr-min', type=float, default=1e-3, help='Min learning rate for training')
	parser.add_argument('--lr-max', type=float, default=1e-2, help='Max learning rate for training')
	parser.add_argument('--batch-size', type=int, default=12, help='Batch size for training')
	parser.add_argument('--size', type=int, default=300, help='Image size for training and validation')
	parser.add_argument('--start-new', type=bool, default=False, help='Start new training instead of resuming')
	parser.add_argument('--test', type=bool, default=False, help='Show a reconstructed test image')
	args = parser.parse_args()
	if args.test:
		test_and_show()
	else:
		train_codec(args.lr_min, args.lr_max, args.batch_size, args.size, args.start_new)
	AppLog.shut_down()
