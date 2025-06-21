import io
import os

import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from src.common.common_utils import AppLog
from src.encoder_decoder.image_reconstruction_loss import VisualInformationFidelityLoss
from src.image_encoder_decoder.image_codec import ImageCodec


class ParquetImageDataset(torch.utils.data.Dataset):
	def __init__(self, parquet_file):
		self.df = pd.read_parquet(parquet_file)
		self.transform = torchvision.transforms.Compose([
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Resize(512, interpolation=InterpolationMode.BILINEAR),
				torchvision.transforms.Normalize(mean=0.5, std=0.5), torchvision.transforms.RandomCrop(512),
		])

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		image_data = self.df.iloc[idx]['image']['bytes']
		image = Image.open(io.BytesIO(image_data))
		return self.transform(image)

def get_data():

	train_set = ParquetImageDataset('data/CC/train.parquet')
	validate_set = ParquetImageDataset('data/CC/validate.parquet')

	train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
	val_loader = DataLoader(validate_set, batch_size=16, shuffle=False)
	return train_loader, val_loader


class TrainEncoderAndDecoder:
	def __init__(self, model, optimizer, train_device, starting_epoch, ending_epoch):
		self.model_orig = model

		self.optimizer = optimizer
		self.device = train_device
		self.current_epoch = starting_epoch
		self.ending_epoch = ending_epoch
		self.best_vloss = float('inf')
		self.model = torch.compile(self.model_orig, mode="default").to(self.device)
		self.vif_loss = VisualInformationFidelityLoss().to(self.device)
		self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=1 / 3, patience=4,
		                                                min_lr=1e-8)

	def train_one_epoch(self, train_loader):
		self.model.train(True)
		tloss = 0.0
		for batch_idx, data in enumerate(train_loader):
			data = data.to(self.device)
			self.optimizer.zero_grad()
			result = self.model(data)
			AppLog.info(f'Batch {batch_idx}: Result: {result.shape} , Data: {data.shape}')
			loss = self.vif_loss.forward(result, data)
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
				vloss += self.vif_loss(result, data).item()
		return vloss

	def train_and_evaluate(self, train_loader, val_loader):

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
			epoch += 1


def prepare_data():
	# Create CC directory if it doesn't exist
	os.makedirs('data/CC', exist_ok=True)

	# Read and concatenate parquet files
	df1 = pd.read_parquet('data/train-00000-of-00002.parquet')
	df2 = pd.read_parquet('data/train-00001-of-00002.parquet')
	combined_df = pd.concat([df1, df2], ignore_index=True)
	AppLog.info(f'Combined dataset size: {len(combined_df)}')
	# Split into train and validation sets (80/20)
	train_size = int(0.8 * len(combined_df))
	train_df = combined_df[:train_size]
	val_df = combined_df[train_size:]

	# Save split datasets
	train_df.to_parquet('data/CC/train.parquet')
	val_df.to_parquet('data/CC/validate.parquet')
	AppLog.info(f'Train dataset size: {len(train_df)}')
	AppLog.info(f'Validation dataset size: {len(val_df)}')


def create_dataset_and_dataloader_from_parquet(parquet_file_path):

	dataset = ParquetImageDataset(parquet_file_path)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
	return dataloader


if __name__ == '__main__':
	# prepare_data()
	torch.set_float32_matmul_precision('high')
	enc = ImageCodec([64, 128, 192, 256], [256, 192, 128, 64, 3])
	optimizer = torch.optim.AdamW(enc.parameters(), lr=1e-4, amsgrad=True)
	traindevice = "cuda" if torch.cuda.is_available() else "cpu"
	train_loader, val_loader = get_data()
	trainer = TrainEncoderAndDecoder(enc, optimizer, traindevice, 0, 30)
	trainer.train_and_evaluate(train_loader, val_loader)
	AppLog.shut_down()
