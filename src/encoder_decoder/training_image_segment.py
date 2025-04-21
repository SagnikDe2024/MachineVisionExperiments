from itertools import groupby

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms.v2.functional import to_dtype
from torchvision.utils import save_image

from src.common.common_utils import AppLog
from src.encoder_decoder.image_segmentation import augment_a_single_image, get_single_channel_max_diff, get_unet


class TrainModel:
	def __init__(self, save_checkpoint_epoch, model, loss_fn, optimizer, device, starting_epoch, ending_epoch) -> None:
		# self.save_checkpoint_epoch = save_checkpoint_epoch
		self.model = model.to(device)
		self.loss_fn = nn.MSELoss()
		self.optimizer = optimizer
		self.device = device
		self.current_epoch = starting_epoch
		self.ending_epoch = ending_epoch
		self.best_vloss = float('inf')

	def train(self, train_loader) -> float:
		self.model.train(True)
		running_loss = 0.0
		train_batch_index = 0
		epoch = self.current_epoch
		EPOCHS = self.ending_epoch
		loss_fn_used = self.loss_fn
		for img in train_loader:
			image, diff_h, diff_w = img[0], img[1], img[2]
			image = image.to(self.device)
			self.optimizer.zero_grad()
			train_batch_index += 1

			segments = self.model.forward(image)
			# AppLog.info(f'Segments shape = {segments.shape}, diff_h = {diff_h.shape}, diff_w = {diff_w.shape}')

			all_loss = self.get_loss(diff_h, diff_w, loss_fn_used, segments)

			all_loss.backward()

			self.optimizer.step()
			running_loss += all_loss.item()
			AppLog.debug(
					f'Epoch [{epoch + 1}/{EPOCHS}]: Batch [{train_batch_index}]: Loss: '
					f'{running_loss / train_batch_index}')
		avg_loss = running_loss / train_batch_index
		return avg_loss

	def get_loss(self, diff_h, diff_w, loss_fn_used, segments):
		(n, c, h, w) = segments.shape
		seg_diff_h_list = []
		seg_diff_w_list = []
		for seg in range(c):
			acquired_segment = segments[:, seg, :, :]
			acquired_segment = acquired_segment.unsqueeze(1)
			seg_diff_w = torch.diff(acquired_segment, dim=-1, prepend=torch.zeros(n, 1, h, 1))
			seg_diff_h = torch.diff(acquired_segment, dim=-2, prepend=torch.zeros(n, 1, 1, w))
			seg_diff_h_list.append(seg_diff_h)
			seg_diff_w_list.append(seg_diff_w)
		all_loss = loss_fn_used(seg_diff_h_list[0], diff_h) + loss_fn_used(seg_diff_w_list[0], diff_w)
		for c in range(1, c):
			all_loss += loss_fn_used(seg_diff_h_list[c], diff_h) + loss_fn_used(seg_diff_w_list[c], diff_w)
		return all_loss

	def evaluate(self, validation_loader) -> float:
		self.model.eval()
		running_vloss = 0.0
		valid_batch_index = 0
		epoch = self.current_epoch
		EPOCHS = self.ending_epoch
		loss_fn_used = self.loss_fn
		with torch.no_grad():
			for img in validation_loader:
				image, diff_h, diff_w = img[0], img[1], img[2]
				image = image.to(self.device)
				segments = self.model.forward(image)
				all_loss = self.get_loss(diff_h, diff_w, loss_fn_used, segments)

				running_vloss += all_loss.item()
				valid_batch_index += 1
				AppLog.debug(
						f'Epoch [{epoch + 1}/{EPOCHS}]: V_Batch [{valid_batch_index}]: V_Loss: '
						f'{running_vloss / valid_batch_index}')
		avg_vloss = running_vloss / valid_batch_index
		return avg_vloss

	def train_and_evaluate(self, train_loader, validation_loader):
		no_improvement = 0
		loss_best_threshold = 1.1

		while self.current_epoch < self.ending_epoch:

			avg_loss = self.train(train_loader)
			AppLog.info(f'Training loss = {avg_loss}')
			avg_vloss = self.evaluate(validation_loader)

			AppLog.info(f'Epoch {self.current_epoch + 1}: Training loss = {avg_loss}, Validation Loss = {avg_vloss}')

			# self.save_checkpoint_epoch(avg_vloss, self.model, self.current_epoch)
			if avg_vloss < self.best_vloss:
				self.best_vloss = avg_vloss
			elif avg_vloss > loss_best_threshold * self.best_vloss:
				AppLog.warning(
						f'Early stopping at {self.current_epoch + 1} epochs as (validation loss = {avg_vloss})/(best '
						f'validation loss = {self.best_vloss}) > {loss_best_threshold} ')
				break
			elif no_improvement > 4:
				AppLog.warning(
						f'Early stopping at {self.current_epoch + 1} epochs as validation loss = {avg_vloss} has '
						f'shown '
						f'no improvement over {no_improvement} epochs')
				break

			else:
				no_improvement += 1

			self.current_epoch += 1

		return self.best_vloss, self.model.model_params


class RawImageDataSet(Dataset):
	def __init__(self, images):
		self.image_files = images

	def __len__(self):
		return len(self.image_files)

	def __getitem__(self, idx):
		images = self.image_files[idx]
		return images


def unsqueeze_image_group(image_group):
	unsqueezed_image_group = [torch.unsqueeze(img_item, 0) for img_item in image_group]
	image_batch = torch.cat(unsqueezed_image_group, dim=0)
	return image_batch


if __name__ == '__main__':
	image = to_dtype(decode_image('data/reddit_face.jpg', mode='RGB'), dtype=torch.float32, scale=True)
	[c, h, w] = image.shape
	images_created = list(augment_a_single_image(image).values())
	grouped = groupby(images_created, lambda x: x[0].shape)
	train_list = []
	for key, group in grouped:
		images_tr, diff_h_tr, diff_w_tr = zip(*group)
		batch_images = unsqueeze_image_group(images_tr)
		diff_h_tr = unsqueeze_image_group(diff_h_tr)
		diff_w_tr = unsqueeze_image_group(diff_w_tr)
		AppLog.info(f'Batch size = {batch_images.shape}')
		AppLog.info(f'Diff h size = {diff_h_tr.shape}')
		AppLog.info(f'Diff w size = {diff_w_tr.shape}')
		train_list.append((batch_images, diff_h_tr, diff_w_tr))

	diff_h, diff_w = get_single_channel_max_diff(image)
	unet = get_unet()
	lr = 0.001
	optimizer = torch.optim.Adam(unet.parameters(), lr=lr)
	compiled_model = unet  #torch.compile(unet, mode="max-autotune")
	train = TrainModel(save_checkpoint_epoch=None, model=compiled_model, loss_fn=None, optimizer=optimizer,
					   device=None,
					   starting_epoch=0, ending_epoch=20)
	image_sq = torch.unsqueeze(image, 0)
	diff_h_sq = torch.unsqueeze(diff_h, 0)
	diff_w_sq = torch.unsqueeze(diff_w, 0)
	AppLog.info(f'Image shape = {image_sq.shape}, diff_h shape = {diff_h_sq.shape}, diff_w shape = {diff_w_sq.shape}')
	model_params = train.train_and_evaluate(train_list, [(image_sq, diff_h_sq, diff_w_sq)])
	segments = compiled_model.forward(torch.unsqueeze(image, 0))
	segmented = torch.cat([segments, torch.zeros([1, 1, h, w])], dim=1).squeeze(0)
	save_image(segmented, 'out/reddit_face_segmented.jpg')
