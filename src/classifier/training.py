import os
import tempfile
from pathlib import Path

import torch
from ray import tune
from ray.tune import Checkpoint
from torch import nn
from torch.optim import lr_scheduler
from torchinfo import summary
from torchvision.datasets import CIFAR10
from torchvision.transforms import InterpolationMode, transforms
from torchvision.transforms.v2 import ColorJitter, Normalize, RandomCrop, RandomHorizontalFlip, RandomInvert, \
	RandomRotation, RandomVerticalFlip

from src.classifier.classifier import Classifier
from src.common.common_utils import AppLog


class ImageAugmentation(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.augment = nn.Sequential(RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5),
		                             RandomInvert(0.5), RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
		                             ColorJitter(brightness=0.5, contrast=1.25, saturation=1.25, hue=0.5),
		                             RandomCrop(32, padding=8),
		                             Normalize((0.5,), (0.5,)))

	def forward(self, x):
		return self.augment(x)


class TrainModel:
	def __init__(self, save_checkpoint_epoch, model, loss_fn, optimizer, device, starting_epoch, ending_epoch) -> None:
		self.augment = torch.compile(ImageAugmentation(), mode="default", dynamic=True).to(device)
		self.save_checkpoint_epoch = save_checkpoint_epoch
		self.model_orig: Classifier = model
		self.loss_fn = loss_fn
		self.optimizer = optimizer
		self.device = device
		self.current_epoch = starting_epoch
		self.ending_epoch = ending_epoch
		self.best_vloss = float('inf')
		self.model = torch.compile(self.model_orig, mode="default")
		self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=1 / 3, patience=4,
		                                                min_lr=1e-8)

	def train(self, train_loader) -> float:
		self.model.train(True)
		running_loss = 0.0
		train_batch_index = 0
		epoch = self.current_epoch
		EPOCHS = self.ending_epoch
		for img, label in train_loader:
			img, label = img.to(self.device), label.to(self.device)
			self.optimizer.zero_grad()
			train_batch_index += 1
			img = self.augment.forward(img)
			raw_prob, prob = self.model.forward(img)
			loss = self.loss_fn(raw_prob, label)
			loss.backward()
			self.optimizer.step()
			running_loss += loss.item()
			AppLog.debug(
					f'Epoch [{epoch + 1}/{EPOCHS}]: Batch [{train_batch_index}]: Loss: '
					f'{running_loss / train_batch_index}')
		avg_loss = running_loss / train_batch_index
		return avg_loss

	def evaluate(self, validation_loader) -> float:
		self.model.eval()
		running_vloss = 0.0
		valid_batch_index = 0
		epoch = self.current_epoch
		EPOCHS = self.ending_epoch
		with torch.no_grad():
			for img, label in validation_loader:
				img, label = img.to(self.device), label.to(self.device)
				raw_prob, prob = self.model.forward(img)
				v_loss = self.loss_fn(raw_prob, label)
				running_vloss += v_loss.item()
				valid_batch_index += 1
				AppLog.debug(
						f'Epoch [{epoch + 1}/{EPOCHS}]: V_Batch [{valid_batch_index}]: V_Loss: '
						f'{running_vloss / valid_batch_index}')
		avg_vloss = running_vloss / valid_batch_index
		return avg_vloss

	def train_and_evaluate(self, train_loader, validation_loader):
		no_improvement = 0
		loss_best_threshold = 1.2

		while self.current_epoch < self.ending_epoch:

			avg_loss = self.train(train_loader)
			avg_vloss = self.evaluate(validation_loader)
			self.scheduler.step(avg_vloss)
			others = [p for name, p in self.model_orig.named_parameters() if 'bias' not in name]
			# AppLog.info(f'There are {others[0]} weight parameters.')
			AppLog.info(
					f'Epoch {self.current_epoch + 1}: Training loss = {avg_loss}, Validation Loss = {avg_vloss}, '
					f'lr = {self.scheduler.get_last_lr()}')

			self.save_checkpoint_epoch(avg_vloss, self.model, self.current_epoch)
			if avg_vloss < self.best_vloss:
				self.best_vloss = avg_vloss
			# elif avg_vloss > loss_best_threshold * self.best_vloss:
			# 	AppLog.warning(
			# 			f'Early stopping at {self.current_epoch + 1} epochs as (validation loss = {avg_vloss})/(best '
			#             f'validation loss = {self.best_vloss}) > {loss_best_threshold} ')
			# 	break
			# elif no_improvement > 9:
			# 	AppLog.warning(
			# 			f'Early stopping at {self.current_epoch + 1} epochs as validation loss = {avg_vloss} has
			# 			shown '
			#             f'no improvement over {no_improvement} epochs')
			# 	break
			#
			# else:
			# 	no_improvement += 1

			self.current_epoch += 1

		return self.best_vloss, self.model.model_params


def save_checkpoint(avg_vloss: float, model: nn.Module, epoch: int) -> None:
	with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
		model_name_temp = f'model_checkpoint.pth'
		torch.save((model.model_params, model.state_dict()),
				   os.path.join(temp_checkpoint_dir, model_name_temp))
		checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
		tune.report({'v_loss': avg_vloss, 'epoch': (epoch + 1)}, checkpoint=checkpoint)


def save_checkpoint_dummy(avg_vloss: float, model: nn.Module, epoch: int) -> None:
	AppLog.info(f'v_loss: {avg_vloss}, epoch: {(epoch + 1)}')


class ExperimentModels:

	def __init__(self, model_creator_func, loader_func, is_ray=True) -> None:
		self.model_creator_func = model_creator_func
		self.loader_func = loader_func
		self.isRay = is_ray

	def execute_single_experiment(self, model_config, batch_size, lr):
		model: nn.Module = self.model_creator_func(model_config)
		batch_size = int(round(batch_size))
		train_loader, validation_loader = self.loader_func(batch_size)
		model_summary = summary(model, input_size=(batch_size, 3, 32, 32))

		optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
		loss_fn = nn.CrossEntropyLoss()
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		AppLog.info(f'Using device: {device}')
		trainable_params = model_summary.trainable_params
		AppLog.info(f'There are {trainable_params} trainable parameters.')

		checkpoint = tune.get_checkpoint() if self.isRay else None
		model = model if self.isRay else model.to(device)
		start = 0
		if checkpoint and self.isRay:
			with checkpoint.as_directory() as checkpoint_dir:
				checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "model_checkpoint.pth"))
				start = checkpoint_dict["epoch"]
				model.load_state_dict(checkpoint_dict["model_state"])
		torch.set_float32_matmul_precision('high')
		checkpoint_save_fn = save_checkpoint if self.isRay else save_checkpoint_dummy
		train_model = TrainModel(checkpoint_save_fn, model, loss_fn, optimizer, device, start, 100)
		best_vloss, model_params = train_model.train_and_evaluate(train_loader, validation_loader)
		AppLog.info(
				f'Best vloss: {best_vloss}, with {trainable_params} params. Performance per param (Higher is better) = '



				f'{1 / (trainable_params * best_vloss)}')
		AppLog.info(f'Classifier best vloss: {best_vloss}, training done. Model params: {model_params}.')
		return {'v_loss': best_vloss, 'trainable_params': trainable_params, 'model_params': model_params}


@torch.compiler.disable(recursive=True)
def load_cifar_dataset(working_dir: Path, batch: int = 500):
	# No need to normalize the training set as the training set is after image augmentation normalized when the model
	# is trained.
	transform = transforms.Compose([transforms.ToTensor()])
	transformV = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
	trainloc: Path = ((working_dir / 'data') / 'CIFAR') / 'train'
	testloc: Path = ((working_dir / 'data') / 'CIFAR') / 'test'

	training_set = CIFAR10(root=trainloc, train=True, download=False, transform=transform)
	validation_set = CIFAR10(root=testloc, train=False, download=False, transform=transformV)
	AppLog.info(f'{len(training_set)} training samples and {len(validation_set)} validation samples')
	train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch, shuffle=True, pin_memory=True,
	                                           drop_last=True)
	validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=5000, shuffle=True, pin_memory=True,
	                                                drop_last=True)

	return train_loader, validation_loader


if __name__ == '__main__':
	classifier = Classifier([256, 256, 10], 32, 2, 54, 192, 7)
	model_summary = summary(classifier, input_size=(100, 3, 32, 32))
	working_dir = Path.cwd()
	loader = lambda batch: load_cifar_dataset(working_dir, int(batch))
	model_creator_f = lambda model_config: classifier
	exp = ExperimentModels(model_creator_f, loader, is_ray=False)
	# torch._dynamo.config.recompile_limit = 64
	exp.execute_single_experiment({'fcn_layers': 3, 'starting_channels': 54, 'cnn_layers': 7, 'final_channels': 256},
	                              100, 1e-5)
