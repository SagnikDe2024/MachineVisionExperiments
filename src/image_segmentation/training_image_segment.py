from itertools import groupby

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms.v2.functional import to_dtype
from torchvision.utils import save_image

from src.common.common_utils import AppLog
from src.image_segmentation.image_segmentation import augment_a_single_image, augment_with_noise, \
	get_single_channel_max_diff, get_unet


def get_boundary(segments, seg_index):
	diff_h_2 = torch.pow(torch.diff(segments[:, seg_index, :, :], dim=-2), 2)
	diff_w_2 = torch.pow(torch.diff(segments[:, seg_index, :, :], dim=-1), 2)
	total_length = diff_h_2.sum() + diff_w_2.sum()

	pass


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

	def infer_and_evaluate_loss(self, image_batch):
		pic, pic_rot_90, pic_rot_180, pic_rot_270 = image_batch

		pic = pic.to(self.device)
		pic_rot_90 = pic_rot_90.to(self.device)
		pic_rot_180 = pic_rot_180.to(self.device)
		pic_rot_270 = pic_rot_270.to(self.device)

		segments = self.model.forward(pic)
		segments_90 = self.model.forward(pic_rot_90)
		segments_180 = self.model.forward(pic_rot_180)
		segments_270 = self.model.forward(pic_rot_270)

		consistency_loss = self.consistency_loss(segments, segments_90, segments_180, segments_270)
		volume_boundary_ratio_loss = self.volume_boundary_ratio_loss(segments)
		volume_boundary_ratio_loss_90 = self.volume_boundary_ratio_loss(segments_90)
		volume_boundary_ratio_loss_180 = self.volume_boundary_ratio_loss(segments_180)
		volume_boundary_ratio_loss_270 = self.volume_boundary_ratio_loss(segments_270)
		volume_boundary_ratio_loss = (
				volume_boundary_ratio_loss + volume_boundary_ratio_loss_90 + volume_boundary_ratio_loss_180 +
				volume_boundary_ratio_loss_270)
		return consistency_loss, volume_boundary_ratio_loss

	def train(self, train_loader) -> float:
		self.model.train(True)
		running_loss = 0.0
		train_batch_index = 0
		epoch = self.current_epoch
		EPOCHS = self.ending_epoch

		for imagen_batch in train_loader:
			self.optimizer.zero_grad()
			consistency_loss, volume_boundary_ratio_loss = self.infer_and_evaluate_loss(imagen_batch)
			all_loss = consistency_loss + volume_boundary_ratio_loss
			all_loss.backward()
			self.optimizer.step()
			running_loss += all_loss.item()
			train_batch_index += 1
			AppLog.info(
					f'Epoch [{epoch + 1}/{EPOCHS}]: Batch [{train_batch_index}]: Loss: '
					f'{running_loss / train_batch_index}, cons_loss: {consistency_loss.item()}, volume_boundary_loss: '
					f'{volume_boundary_ratio_loss.item()}')
		avg_loss = running_loss / train_batch_index
		return avg_loss

	def get_loss2(self, segments):
		diff_h_2 = torch.diff(segments, dim=-2)
		diff_w_2 = torch.diff(segments, dim=-1)
		return self.loss_fn(0, diff_h_2, reduce='sum') + self.loss_fn(0, diff_w_2, reduce='sum')

	def volume_boundary_ratio_loss(self, segments):
		segment_volumes = segments.sum(dim=(-2, -1))
		boundary_length_x = torch.pow(torch.diff(segments, dim=-1), 2)
		boundary_length_y = torch.pow(torch.diff(segments, dim=-2), 2)
		diag_length = torch.pow(boundary_length_x + boundary_length_y, 1 / 2)
		total_length = diag_length.sum(dim=(-2, -1))
		length_segment_volume_ratio = total_length / segment_volumes
		return self.loss_fn(length_segment_volume_ratio, 0, reduce='sum')

	def consistency_loss(self, segments, segment_90, segment_180, segment_270):
		segments_90_unrot = torch.rot90(segment_90, -1, [-1, -2])
		segments_180_unrot = torch.rot90(segment_180, -2, [-1, -2])
		segments_270_unrot = torch.rot90(segment_270, -3, [-1, -2])
		return (self.loss_fn(segments, segments_90_unrot, reduction='sum') +
				self.loss_fn(segments_90_unrot, segments_180_unrot, reduction='sum') +
				self.loss_fn(segments_180_unrot, segments_270_unrot, reduction='sum') +
				self.loss_fn(segments_270_unrot, segments, reduction='sum'))

	def get_loss(self, diff_h, diff_w, loss_fn_used, segments):
		(n, c, h, w) = segments.shape
		seg_diff_h_list = []
		seg_diff_w_list = []
		for seg in range(c):
			acquired_segment = segments[:, seg, :, :]
			acquired_segment = acquired_segment.unsqueeze(1)
			seg_diff_w = torch.diff(acquired_segment, dim=-1)[:, :, 1:h, :]
			seg_diff_h = torch.diff(acquired_segment, dim=-2)[:, :, :, 1:w]
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
				consistency_loss, volume_boundary_ratio_loss = self.infer_and_evaluate_loss(img)

				all_loss = consistency_loss + volume_boundary_ratio_loss

				running_vloss += all_loss.item()
				valid_batch_index += 1
				AppLog.info(
						f'Epoch [{epoch + 1}/{EPOCHS}]: V_Batch [{valid_batch_index}]: V_Loss: '
						f'{running_vloss / valid_batch_index}, cons_loss: {consistency_loss.item()}, '
						f'volume_boundary_loss: {volume_boundary_ratio_loss.item()}')
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


def the_main():

	image = to_dtype(decode_image('data/reddit_face.jpg', mode='RGB'), dtype=torch.float32, scale=True)
	[c, h, w] = image.shape
	images_created = list(augment_a_single_image(image).values())
	grouped = groupby(images_created, lambda x: x[0].shape)
	train_list = []
	diff_h, diff_w = get_single_channel_max_diff(image)
	save_diff = torch.cat([diff_h, diff_w, torch.zeros([1, h - 1, w - 1])], dim=0)
	save_image(save_diff, 'out/reddit_face_diff.jpg')
	# return

	for key, group in grouped:
		images_tr, diff_h_tr, diff_w_tr = zip(*group)
		batch_images = unsqueeze_image_group(images_tr)
		diff_h_tr = unsqueeze_image_group(diff_h_tr)
		diff_w_tr = unsqueeze_image_group(diff_w_tr)
		AppLog.info(f'Batch size = {batch_images.shape}')
		AppLog.info(f'Diff h size = {diff_h_tr.shape}')
		AppLog.info(f'Diff w size = {diff_w_tr.shape}')
		train_list.append((batch_images, diff_h_tr, diff_w_tr))
	unet = get_unet()
	lr = 0.001
	optimizer = torch.optim.Adam(unet.parameters(), lr=lr)
	compiled_model = unet  # torch.compile(unet, mode="reduce-overhead", fullgraph=True)
	train = TrainModel(save_checkpoint_epoch=None, model=compiled_model, loss_fn=None, optimizer=optimizer,
					   device=None,
					   starting_epoch=0, ending_epoch=20)
	image_sq = torch.unsqueeze(image, 0)
	diff_h_sq = torch.unsqueeze(diff_h, 0)
	diff_w_sq = torch.unsqueeze(diff_w, 0)
	AppLog.info(f'Image shape = {image_sq.shape}, diff_h shape = {diff_h_sq.shape}, diff_w shape = {diff_w_sq.shape}')
	model_params = train.train_and_evaluate(train_list, [(image_sq, diff_h_sq, diff_w_sq)])
	torch.save((model_params, unet.state_dict()), 'models/segment_model_params.pt')

	eval = compiled_model.forward(torch.unsqueeze(image, 0))
	# segments = torch.argmax(eval, dim=1)
	# one_hot_ver = torch.nn.functional.one_hot(segments, num_classes=eval.shape[1])
	# converted = torch.permute(one_hot_ver, (0, 3, 1, 2))

	segmented = torch.cat([eval, torch.zeros([1, 1, h, w])], dim=1).squeeze(0)
	save_image(segmented, 'out/reddit_face_segmented_train.jpg')


def evaluate_model():
	unet = get_unet()
	model_params, unet_state = torch.load('models/segment_model_params.pt')
	unet.load_state_dict(unet_state)
	AppLog.info(f'Unet loaded')
	image = to_dtype(decode_image('data/reddit_face.jpg', mode='RGB'), dtype=torch.float32, scale=True)
	[c, h, w] = image.shape
	unet.eval()
	with torch.no_grad():
		eval = unet.forward(torch.unsqueeze(image, 0)).squeeze(0)
		AppLog.info(f'The model has evaluated the image')
	# segments = torch.argmax(eval, dim=0)
	# segments+=1

	# one_hot = torch.nn.functional.one_hot(segments, num_classes=eval.shape[0])

	# one_hot = torch.permute(one_hot, (2, 0, 1))

	segmented = torch.cat([eval, torch.zeros([1, h, w])], dim=0)
	save_image(segmented, 'out/reddit_face_segmented.jpg')


def acquire_image(image_path):
	return to_dtype(decode_image(image_path, mode='RGB'), dtype=torch.float32, scale=True)


def check_transformations(batch_size=32):
	image = to_dtype(decode_image('data/reddit_face.jpg', mode='RGB'), dtype=torch.float32, scale=True)
	[c, h, w] = image.shape
	images_created = augment_with_noise(image, batch_size)
	orig = images_created[0].unsqueeze(0)
	noised = images_created[1:]
	batched = torch.stack(noised)
	return orig, batched


def get_rotated_images(image, batch_size=32):
	rot_90 = torch.rot90(image, 1, [-1, -2])
	rot_180 = torch.rot90(image, 2, [-1, -2])
	rot_270 = torch.rot90(image, 3, [-1, -2])
	orig, images_created = augment_with_noise(image, batch_size)
	rot_90, images_created_90 = augment_with_noise(rot_90, batch_size)
	rot_180, images_created_180 = augment_with_noise(rot_180, batch_size)
	rot_270, images_created_270 = augment_with_noise(rot_270, batch_size)
	return orig, images_created, images_created_90, images_created_180, images_created_270


def train_model_part_2():
	image = acquire_image('data/reddit_face.jpg')
	[c, h, w] = image.shape

	batches = 20
	train_list = []
	for _ in range(batches):
		orig, batched, batched_90, batched_180, batched_270 = get_rotated_images(image)
		train_list.append((batched, batched_90, batched_180, batched_270))

	unet = get_unet()
	lr = 0.001
	optimizer = torch.optim.Adam(unet.parameters(), lr=lr)
	compiled_model = unet  # torch.compile(unet, mode="reduce-overhead", fullgraph=True)
	train = TrainModel(save_checkpoint_epoch=None, model=compiled_model, loss_fn=None, optimizer=optimizer,
					   device=None,
					   starting_epoch=0, ending_epoch=20)
	AppLog.info(f'Image shape = {image.shape}, ')
	image_sq = torch.unsqueeze(image, 0)
	rot_90_sq = torch.rot90(image_sq, 1, [-1, -2])
	rot_180_sq = torch.rot90(image_sq, 2, [-1, -2])
	rot_270_sq = torch.rot90(image_sq, 3, [-1, -2])
	model_params = train.train_and_evaluate(train_list, [(image_sq, rot_90_sq, rot_180_sq, rot_270_sq)])


if __name__ == '__main__':
	# the_main()
	# evaluate_model()
	check_transformations()
