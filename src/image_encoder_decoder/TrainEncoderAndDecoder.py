import torch
import torchvision
from torch.cuda import device
from torch.optim import lr_scheduler
from torchvision.datasets import cityscapes

from src.common.common_utils import AppLog
from src.encoder_decoder.image_reconstruction_loss import VisualInformationFidelityLoss
from src.image_encoder_decoder.image_codec import ImageCodec


def get_data():
	transform = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=0.5, std=0.5),
	])

	train_data = cityscapes.Cityscapes('data/Cityscape', "train", mode='fine',target_type='semantic', transform=transform)
	val_data = cityscapes.Cityscapes('data/Cityscape', "val", mode='fine',target_type='semantic', transform=transform)

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
	val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)
	return train_loader, val_loader



class TrainEncoderAndDecoder:
	def __init__(self, model,optimizer, train_device, starting_epoch, ending_epoch):
		self.model_orig = model
		self.vif_loss = VisualInformationFidelityLoss()
		self.optimizer = optimizer
		self.device = train_device
		self.current_epoch = starting_epoch
		self.ending_epoch = ending_epoch
		self.best_vloss = float('inf')
		self.model = torch.compile(self.model_orig, mode="default")
		self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=1 / 3, patience=4,
		                                                min_lr=1e-8)

	def train_one_epoch(self,train_loader):
		self.model.train(True)
		tloss = 0.0
		for batch_idx, (data, target) in enumerate(train_loader):
			data= data.to(device)
			self.optimizer.zero_grad()
			result = self.model(data)
			loss = self.vif_loss(result,data)
			tloss += loss.item()
			loss.backward()
			self.optimizer.step()
		return tloss

	def evaluate(self,val_loader):
		self.model.eval()
		vloss = 0.0
		for batch_idx, (data, target) in enumerate(val_loader):
			data= data.to(device)
			result = self.model(data)
			vloss += self.vif_loss(result,data).item()
		return vloss

	def train_and_evaluate(self,train_loader,val_loader):

		epoch = self.current_epoch
		while epoch < self.ending_epoch:
			train_loss = self.train_one_epoch(train_loader)
			val_loss = self.evaluate(val_loader)
			self.scheduler.step(val_loss)
			AppLog.info(f'Epoch {epoch + 1}: Training loss = {train_loss}, Validation Loss = {val_loss}, lr = {self.scheduler.get_last_lr()}')
			if val_loss < self.best_vloss:
				self.best_vloss = val_loss
			epoch += 1


if __name__ == '__main__':
	enc = ImageCodec([64, 128, 192, 256], [256, 192, 128, 64, 3])
	optimizer = torch.optim.AdamW(enc.parameters(), lr=1e-4, amsgrad=True)
	traindevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	train_loader, val_loader = get_data()
	trainer = TrainEncoderAndDecoder(enc, optimizer, traindevice, 0, 30)
	trainer.train_and_evaluate(train_loader, val_loader)












