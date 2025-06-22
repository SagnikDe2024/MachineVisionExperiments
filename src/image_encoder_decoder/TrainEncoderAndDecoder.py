import io
import os
from pathlib import Path

import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode

from src.common.common_utils import AppLog
from src.encoder_decoder.image_reconstruction_loss import MultiscalePerceptualLoss
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
		image = Image.open(image_path)
		return self.transform(image)

def get_data():
	size = 300
	transform = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Resize(size, interpolation=InterpolationMode.BILINEAR),
			torchvision.transforms.Normalize(mean=0.5, std=0.5), torchvision.transforms.RandomCrop(size),
	])

	train_set = ImageFolderDataset(Path('data/CC/train'), transform=transform)
	validate_set = ImageFolderDataset(Path('data/CC/validate'), transform=transform)

	train_loader = DataLoader(train_set, batch_size=12, shuffle=True,drop_last=True)
	val_loader = DataLoader(validate_set, batch_size=12, shuffle=False,drop_last=True)
	return train_loader, val_loader


class TrainEncoderAndDecoder:
	def __init__(self, model, optimizer, train_device, starting_epoch, ending_epoch, vloss=float('inf')):
		self.model_orig = model

		self.optimizer = optimizer
		self.device = train_device
		self.current_epoch = starting_epoch
		self.ending_epoch = ending_epoch
		self.best_vloss = vloss
		self.model = torch.compile(self.model_orig, mode="default").to(self.device)
		self.loss_func = MultiscalePerceptualLoss().to(self.device)
		self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=1 / 3, patience=4,
		                                                min_lr=1e-8)


	def train_one_epoch(self, train_loader):
		self.model.train(True)
		tloss = 0.0

		for batch_idx, data in enumerate(train_loader):
			data = data.to(self.device)
			self.optimizer.zero_grad()
			result = self.model(data)
			loss = self.loss_func.forward(result, data)
			tloss += loss.item()
			loss.backward()
			self.optimizer.step()
		return tloss

	def evaluate(self, val_loader):
		self.model.eval()
		vloss = 0.0
		with torch.no_grad():
			for batch_idx, data, in enumerate(val_loader):
				data = data.to(self.device)
				result = self.model(data)
				vloss += self.loss_func(result, data).item()
		return vloss

	def train_and_evaluate(self, train_loader, val_loader):
		AppLog.info(f'Training from {self.current_epoch} to {self.ending_epoch} epochs.')
		epoch = self.current_epoch
		while epoch < self.ending_epoch:
			train_loss = self.train_one_epoch(train_loader)
			val_loss = self.evaluate(val_loader)
			self.scheduler.step(val_loss)
			AppLog.info(
					f'Epoch {epoch + 1}: Training loss = {train_loss}, Validation Loss = {val_loss}, '
					f'lr = {self.scheduler.get_last_lr()}')
			if val_loss < self.best_vloss:
				self.best_vloss = val_loss
				save_training_state('checkpoints/encode_decode/train_codec.pth', self.model_orig.state_dict(),
				                    self.optimizer.state_dict(), epoch, val_loss)
			epoch += 1


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
			'v_loss'              : vloss, }, location, )


def load_training_state(location, model, optimizer):
	checkpoint = torch.load(location)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	vloss = checkpoint['v_loss']
	return model, optimizer, epoch, vloss


def train_codec():
	save_location = 'checkpoints/encode_decode/train_codec.pth'
	enc = ImageCodec([64, 128, 192, 256], [256, 192, 128, 64, 3])
	optimizer = torch.optim.AdamW(enc.parameters(), lr=1e-3, amsgrad=True)
	traindevice = "cuda" if torch.cuda.is_available() else "cpu"
	train_loader, val_loader = get_data()
	if os.path.exists(save_location):
		enc, optimizer, epoch, vloss = load_training_state(save_location, enc, optimizer)
		AppLog.info(f'Loaded checkpoint from epoch {epoch}')
		trainer = TrainEncoderAndDecoder(enc, optimizer, traindevice, epoch, 30, vloss)
		trainer.train_and_evaluate(train_loader, val_loader)
	else:
		trainer = TrainEncoderAndDecoder(enc, optimizer, traindevice, 0, 30)
		trainer.train_and_evaluate(train_loader, val_loader)


if __name__ == '__main__':
	# prepare_data()
	train_codec()
	AppLog.shut_down()
