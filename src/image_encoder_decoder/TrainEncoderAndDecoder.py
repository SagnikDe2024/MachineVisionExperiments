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
from torchvision.transforms.v2 import CenterCrop, RandomCrop, RandomHorizontalFlip, RandomResize, RandomVerticalFlip, \
	Resize

from src.common.common_utils import AppLog, acquire_image
from src.encoder_decoder.image_reconstruction_loss import ReconstructionLoss
from src.image_encoder_decoder.image_codec import ImageCodec


class ImageFolderDataset(Dataset):
	def __init__(self, path : Path,transform=None):
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

def get_data():
	minsize = 256
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

	train_loader = DataLoader(train_set, batch_size=12, shuffle=True,drop_last=True)
	val_loader = DataLoader(validate_set, batch_size=12, shuffle=False,drop_last=True)
	return train_loader, val_loader


class TrainEncoderAndDecoder:
	def __init__(self, model, optimizer, train_device, cycle_sch, save_training_fn, starting_epoch, ending_epoch,
	             vloss=float('inf')):
		self.model_orig = model
		self.save_training_fn = save_training_fn
		self.optimizer = optimizer
		self.device = train_device
		self.current_epoch = starting_epoch
		self.ending_epoch = ending_epoch
		self.best_vloss = vloss
		self.model = torch.compile(self.model_orig, mode="default").to(self.device)
		self.trained_one_batch = False
		# self.loss_func = torch.compile(MultiscalePerceptualLoss(max_downsample=4), mode="default").to(self.device)
		self.loss_func = torch.compile(ReconstructionLoss(), mode="default").to(self.device)
		self.scheduler = cycle_sch(self.optimizer)

	# self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=1 / 3, patience=3,
	#                                                 min_lr=1e-8)


	def train_one_epoch(self, train_loader):
		self.model.train(True)
		tloss = 0.0
		pics_seen = 0
		for batch_idx, data in enumerate(train_loader):
			data = data.to(self.device)
			data = (data - 1 / 2) * 2
			self.optimizer.zero_grad()
			result = self.model(data)
			result = result / 2 + 1 / 2
			loss = self.loss_func.forward(result, data)
			tloss += loss.item()
			pics_seen += data.shape[0]
			loss.backward()
			self.optimizer.step()
			self.scheduler.step()
			if not self.trained_one_batch:
				self.trained_one_batch = True
				AppLog.info(f'Training loss: {tloss}, batch: {batch_idx + 1}')
		return tloss/pics_seen

	def evaluate(self, val_loader):
		self.model.eval()
		vloss = 0.0
		pics_seen = 0
		with torch.no_grad():
			for batch_idx, data, in enumerate(val_loader):
				data = data.to(self.device)
				data = (data - 1 / 2) * 2
				result = self.model(data)
				result = result / 2 + 1 / 2
				vloss += self.loss_func(result, data).item()
				pics_seen += data.shape[0]
		return vloss/pics_seen

	def train_and_evaluate(self, train_loader, val_loader):
		AppLog.info(f'Training from {self.current_epoch} to {self.ending_epoch} epochs.')
		while self.current_epoch < self.ending_epoch:
			train_loss = self.train_one_epoch(train_loader)
			val_loss = self.evaluate(val_loader)
			# self.scheduler.step(val_loss,epoch=epoch)
			AppLog.info(
					f'Epoch {self.current_epoch + 1}: Training loss = {train_loss:.3e}, Validation Loss = {val_loss:.3e}, '
					f'lr = {self.scheduler.get_last_lr()}')
			if val_loss < self.best_vloss:
				self.best_vloss = val_loss
				self.save_training_fn(self.model_orig, self.optimizer, self.current_epoch, val_loss)
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
	for i,row in enumerate(combined_df.iterrows()):
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
	            'v_loss': vloss, }, location, )


def load_training_state(location, model, optimizer):
	checkpoint = torch.load(location)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	vloss = checkpoint['v_loss']
	return model, optimizer, epoch, vloss


def train_codec(lr_min, lr_max, start_new):
	save_location = 'checkpoints/encode_decode/train_codec_augmented.pth'
	enc = ImageCodec([64, 128, 192, 256], [256, 192, 128, 64])
	optimizer = torch.optim.AdamW(
			[{'params': enc.encoder.parameters()},
			 {'params': enc.decoder.parameters(), 'weight_decay': 0.0001}],
			lr=lr_min)
	traindevice = "cuda" if torch.cuda.is_available() else "cpu"
	train_loader, val_loader = get_data()
	save_training_fn = lambda enc_p, optimizer_p, epoch_p, vloss_p: save_training_state(save_location, enc_p,
	                                                                                    optimizer_p, epoch_p, vloss_p)

	if os.path.exists(save_location) and not start_new:
		enc, optimizer, epoch, vloss = load_training_state(save_location, enc, optimizer)
		AppLog.info(f'Loaded checkpoint from epoch {epoch} with vloss {vloss:.3e}')
		scheduler_fn = lambda optim : CyclicLR(optim, base_lr=lr_min, max_lr=lr_max, mode='triangular2')
		trainer = TrainEncoderAndDecoder(enc, optimizer, traindevice, scheduler_fn, save_training_fn, epoch, 30, vloss)
		trainer.train_and_evaluate(train_loader, val_loader)
	else:
		AppLog.info(f'Training from scratch. Using learning rate {lr_min} and device {traindevice}')
		scheduler_fn = lambda optim : CyclicLR(optim, base_lr=lr_min, max_lr=lr_max, mode='triangular2')
		trainer = TrainEncoderAndDecoder(enc, optimizer, traindevice, scheduler_fn, save_training_fn, 0, 30)
		trainer.train_and_evaluate(train_loader, val_loader)


def test_and_show():
	save_location = 'checkpoints/encode_decode/train_codec_augmented.pth'
	enc = ImageCodec([64, 128, 192, 256], [256, 192, 128, 64])
	optimizer = torch.optim.SGD(enc.parameters(), lr=0.1)
	traindevice = "cuda" if torch.cuda.is_available() else "cpu"
	if os.path.exists(save_location):
		enc, optimizer, epoch, vloss = load_training_state(save_location, enc, optimizer)
		AppLog.info(f'Loaded checkpoint from epoch {epoch}')
		enc.eval()
		enc.to(traindevice)
		with torch.no_grad():
			image = acquire_image('data/normal_pic.jpg')
			image = image.unsqueeze(0)
			image = image.to(traindevice)
			imagef = (image - 1 / 2) * 2
			encoded = enc.forward(imagef)
			encoded = (encoded / 2) + 1 / 2
			image_pil = torchvision.transforms.ToPILImage()(image.squeeze(0))
			encoded_pil = torchvision.transforms.ToPILImage()(encoded.squeeze(0))
			image_pil.show()
			encoded_pil.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train encoder and decoder model')
	parser.add_argument('--lr-min', type=float, default=1e-3, help='Min learning rate for training')
	parser.add_argument('--lr-max', type=float, default=1e-2, help='Max learning rate for training')
	parser.add_argument('--start-new', type=bool, default=False, help='Start new training instead of resuming')
	args = parser.parse_args()
	# prepare_data()
	train_codec(args.lr_min,args.lr_max, args.start_new)
	# test_and_show()
	AppLog.shut_down()
