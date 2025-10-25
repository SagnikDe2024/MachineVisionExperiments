from random import Random

import torch
from torch import nn
from torch.nn import SmoothL1Loss
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.image import VisualInformationFidelity
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, RandomVerticalFlip

from src.common.common_utils import AppLog
from src.common.image_delentropy import SimpleDenseLayer
from src.image_cleanup.image_defects import add_jpeg_artifacts
from src.image_encoder_decoder.TrainEncoderAndDecoder import get_data, save_training_state
from src.image_encoder_decoder.image_codec import calc_channels_depth_and_midchs


class SimilarityNetwork(nn.Module):
	def __init__(self, ch_in_enc, ch_out_enc, layers=5, out_groups=16, min_depth=16, max_depth=16, min_mid_ch=12):
		super().__init__()
		self.min_depth = min_depth
		self.max_depth = max_depth
		channels, depths, mid_chs, layer_group = calc_channels_depth_and_midchs(ch_in_enc, ch_out_enc, self.min_depth,
		                                                                        self.max_depth, layers,
		                                                                        mid_ch_st=min_mid_ch)
		AppLog.info(f'Encoder channels {channels}, depths {depths}, mid_chs {mid_chs}, groups {layer_group}')
		dropout = [0.1 + 0.4 * i / (layers - 1) for i in range(layers)]

		all_layers = []
		for l_i in range(layers):

			if l_i == 0:
				dense_layer = SimpleDenseLayer(channels[l_i], mid_chs[l_i], channels[l_i + 1], depths[l_i],
				                               layer_group[l_i], dropped_out=dropout[l_i])
			elif l_i < layers - 1:
				dense_layer = SimpleDenseLayer(channels[l_i], mid_chs[l_i], channels[l_i + 1], depths[l_i],
				                               layer_group[l_i], out_groups=layer_group[l_i], dropped_out=dropout[l_i])
			else:
				dense_layer = SimpleDenseLayer(channels[l_i], mid_chs[l_i], channels[l_i + 1], depths[l_i], out_groups,
				                               in_groups=4, out_groups=out_groups, dropped_out=dropout[l_i])
			all_layers.append(dense_layer)
			if l_i < layers - 1:
				all_layers.append(nn.AvgPool2d(2))
			self.ch_out = dense_layer.out_ch

		self.final_downsample = nn.AdaptiveAvgPool2d(1)
		all_layers.append(self.final_downsample)
		self.all_layers_list = all_layers
		self.encoder_layers = nn.Sequential(*self.all_layers_list)
		self.layer_count = layers
		self.simcalc = nn.CosineSimilarity(dim=1)

	def forward(self, inf_img, tar_img):
		inf_img = (inf_img - 0.5) / 0.5
		tar_img = (tar_img - 0.5) / 0.5
		inf_embed = self.encoder_layers(inf_img)
		tar_embed = self.encoder_layers(tar_img)
		similarity = self.simcalc(inf_embed, tar_embed)
		mean_similarity = torch.mean(similarity)
		# reshaped = torch.reshape(similarity,[1])
		return mean_similarity


class TrainSimilarityModel:
	def __init__(self, model, optimizer, train_device, cycle_sch, save_training_fn, starting_epoch, ending_epoch,
	             vloss=float('inf')):

		self.random = Random()
		self.device = train_device
		self.model = model.to(self.device)
		self.save_training_fn = save_training_fn
		self.optimizer = optimizer

		self.current_epoch = starting_epoch
		self.ending_epoch = ending_epoch
		self.best_vloss = vloss

		self.trained_one_batch = False

		self.loss_func = SmoothL1Loss(beta=1)
		self.scheduler = cycle_sch
		self.train_transform = Compose([RandomVerticalFlip(0.5), RandomHorizontalFlip(0.5)])
		self.similarity_model = VisualInformationFidelity()

	def train_compilable(self, data, ratio) -> torch.Tensor:
		data = self.train_transform(data)
		smooth_loss = self.common_step(data, ratio)
		return smooth_loss

	def validate_compiled(self, data, ratio) -> torch.Tensor:
		smooth_loss = self.common_step(data, ratio)
		return smooth_loss

	def common_step(self, data, ratio):
		percent = round(ratio * 100)
		clamped_percent = min(max(percent, 1), 99)
		defects = add_jpeg_artifacts(data, clamped_percent)
		vif_score = self.similarity_model(defects, data)
		loss = self.compilable_part(data, defects, vif_score)
		return loss

	@torch.compile(mode='max-autotune')
	def compilable_part(self, data, defects: torch.Tensor, vif_score) -> torch.Tensor:
		similarity = self.model(defects, data)
		loss = self.loss_func(similarity, vif_score)
		return loss

	def train_one_epoch(self, train_loader):
		self.model.train(True)
		t_loss = {'smooth_loss': 0.0, }
		for batch_idx, data in enumerate(train_loader):
			ratio = self.random.random() * 0.99 + 0.005
			self.optimizer.zero_grad()
			data = data.to(self.device)
			loss = self.train_compilable(data, ratio)
			loss_scalar = loss.item()
			if loss_scalar != loss_scalar:
				AppLog.info(f'Nan detected @ batch: {batch_idx + 1}')
				self.scheduler.step()
				continue

			loss.backward()
			self.optimizer_step_compile()
			self.scheduler.step()
			t_loss['smooth_loss'] += loss_scalar
			if not self.trained_one_batch:
				self.trained_one_batch = True
				AppLog.info(f'Training loss: {t_loss}, batch: {batch_idx + 1}')
		batches = len(train_loader)
		t_loss['smooth_loss'] /= batches
		return t_loss

	@torch.compile(fullgraph=False)
	def optimizer_step_compile(self):
		self.optimizer.step()

	def evaluate(self, val_loader):
		self.model.eval()
		vloss = {'smooth_loss': 0.0, }
		with torch.no_grad():
			for batch_idx, data, in enumerate(val_loader):
				ratio = self.random.random() * 0.98 + 0.01

				# transformed = self.validate_transform(data)
				# stacked = torch.stack(transformed)
				stacked = data
				stacked = stacked.to(self.device)
				s, n, c, h, w = stacked.shape
				reshaped = torch.reshape(stacked, (s * n, c, h, w))
				smooth_loss1 = self.validate_compiled(reshaped, ratio)
				loss_scalar = smooth_loss1.item()
				if loss_scalar != loss_scalar:
					AppLog.info(f'Nan detected @ batch: {batch_idx + 1}')
					continue

				vloss['smooth_loss'] += smooth_loss1.item()

		batches = len(val_loader)
		vloss['smooth_loss'] /= batches

		return vloss

	def train_and_evaluate(self, train_loader, val_loader):

		AppLog.info(f'Training from {self.current_epoch} to {self.ending_epoch} epochs.')
		while self.current_epoch < self.ending_epoch:
			train_loss = self.train_one_epoch(train_loader)
			val_loss_dic = self.evaluate(val_loader)
			AppLog.info(
				f'Epoch {self.current_epoch + 1}: Training loss = {train_loss} , Validation Loss = {val_loss_dic} lr '
				f'= {(self.scheduler.get_last_lr()[0]):.3e} ')
			val_loss = val_loss_dic['smooth_loss']
			if val_loss < self.best_vloss:
				self.best_vloss = val_loss
				self.save_training_fn(self.model, self.optimizer, self.current_epoch + 1, val_loss,
				                      self.scheduler)
			self.current_epoch += 1


def train_codec(lr_min_arg, lr_max_arg, batch_size, size, epochs, reset_vloss, start_new):
	save_location = 'checkpoints/similarity/trained_similarity.pth'
	traindevice = "cuda" if torch.cuda.is_available() else "cpu"

	lr_min = lr_min_arg if lr_min_arg > 0 else 1e-8
	lr_max = lr_max_arg if lr_max_arg > 0 else 1e-2

	similarty_model = SimilarityNetwork(64, 768)
	decay_params = []
	no_decay_params = []
	for name, param in similarty_model.named_parameters():
		if not param.requires_grad:
			continue
		# Apply weight decay only to convolutional layers, not to normalization layers
		if 'GroupNorm' in name or 'norm' in name or 'bn' in name or 'bias' in name:
			no_decay_params.append(param)
		else:
			decay_params.append(param)

	optimizer = torch.optim.AdamW(
		[{'params': decay_params, 'weight_decay': 1e-4}, {'params': no_decay_params, 'weight_decay': 0}], lr=lr_min,
		fused=True)
	max_epochs = epochs
	train_loader, val_loader = get_data(batch_size=batch_size, minsize=size)
	save_training_fn = lambda enc_p, optimizer_p, epoch_p, vloss_p, sch: save_training_state(save_location, enc_p,
	                                                                                         optimizer_p, epoch_p,
	                                                                                         vloss_p, sch, None)
	cyc_sch = OneCycleLR(optimizer, max_lr=lr_max, epochs=max_epochs, steps_per_epoch=len(train_loader))

	# if os.path.exists(save_location) and not start_new:
	# 	if lr_max_arg > 0 and lr_min_arg > 0:
	# 		enc, optim, epoch, vloss, scheduler, scaler = load_training_state(save_location, enc, None, None, None)
	# 		scheduler = cyc_sch
	# 		optim = optimizer
	# 	else:
	# 		enc, optim, epoch, vloss, scheduler, scaler = load_training_state(save_location, enc, optimizer, cyc_sch,
	# 		                                                                  GradScaler())
	# 	AppLog.info(f'Loaded checkpoint from epoch {epoch} with vloss {vloss:.3e} and scheduler {scheduler}')
	# 	if reset_vloss:
	# 		vloss = float('inf')
	# 		epoch = 0
	# 	AppLog.info(f'(Re)Starting from epoch {epoch} with vloss {vloss:.3e} and scheduler {scheduler}, using device '
	# 	            f'{traindevice}')
	#
	# 	trainer = TrainEncoderAndDecoder(enc, optim, traindevice, scheduler, save_training_fn, epoch, max_epochs,
	# 	                                 vloss,
	# 	                                 scaler)
	# 	trainer.train_and_evaluate(train_loader, val_loader)
	# else:
	# 	AppLog.info(
	# 		f'Training from scratch. Using lr_min={lr_min}, lr_max={lr_max} and scheduler {cyc_sch}, using device '
	# 		f'{traindevice}')
	trainer = TrainSimilarityModel(similarty_model, optimizer, traindevice, cyc_sch, save_training_fn, 0, max_epochs)
	trainer.train_and_evaluate(train_loader, val_loader)


if __name__ == '__main__':
	train_codec(lr_min_arg=1e-8, lr_max_arg=1e-2, batch_size=10, size=287, epochs=50, reset_vloss=False,
	            start_new=False)
# vifs = VisualInformationFidelity(reduction='none')
#
# # sim = SimilarityNetwork(64, 768)
# inp1 = torch.randn(8, 3, 832, 232)
# inp2 = torch.randn(8, 3, 832, 232)
# vif = vifs(inp1,inp2)
# print(vif)
# res = sim(inp1,inp2)
# summary(sim, input_data=[inp1, inp2])
# print(res)
