import os
from datetime import datetime
from pathlib import Path

import numpy as np
import ray
import torch
from ray import tune
from ray.tune import Result, RunConfig, Tuner
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from src.classifier.classifier import Classifier
from src.utils.common_utils import AppLog
from src.wip.training import ExperimentModels


def load_cifar_dataset(working_dir, batch: int = 500):
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

	trainloc: Path = working_dir / 'data' / 'CIFAR' / 'train'
	testloc: Path = working_dir / 'data' / 'CIFAR' / 'test'

	training_set = CIFAR10(root=trainloc, train=True, download=False, transform=transform)
	validation_set = CIFAR10(root=testloc, train=False, download=False, transform=transform)
	AppLog.info(f'{len(training_set)} training samples and {len(validation_set)} validation samples')
	train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch, shuffle=True, pin_memory=True,
											   drop_last=True)
	validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=5000, shuffle=True, pin_memory=True,
													drop_last=True)

	return train_loader, validation_loader

class TuneClassifier:
	def __init__(self, working_dir: Path, samples):
		self.dir_num = 0
		self.working_dir = working_dir
		self.checkpoint_store = self.working_dir / 'checkpoints'
		scheduler = ASHAScheduler(metric='v_loss', mode='min', time_attr='epoch', max_t=40, grace_period=5,
								  reduction_factor=3)
		search = OptunaSearch(metric='v_loss', mode='min')
		self.tune_run_config = RunConfig(
				name='tune_classifier',
				storage_path=self.checkpoint_store,
		)
		experiment = ExperimentModels(create_classifier_from_config, load_cifar_dataset)
		tune_exp = lambda tune_params: tune_with_exp(experiment, tune_params)
		self.search_space = {'learning_rate'    : tune.loguniform(0.00001, 0.0075),
							 'dnn_layers'       : tune.quniform(4, 7, 1),
							 'final_size'       : tune.quniform(1, 4, 1),
							 'starting_channels': tune.qloguniform(12, 32, 4),
							 'final_channels'   : tune.sample_from(
									 lambda spec: np.random.uniform(256 / (spec.config.final_size ** 2),
																	2048 / (spec.config.final_size ** 2))),
							 'cnn_layers'       : tune.quniform(3, 6, 1),
							 'batch_size'       : tune.choice([100, 125, 250])}

		self.trainable_with_resources = tune.with_resources(tune_exp, {'cpu': 1, "gpu": 0.45})
		self.tune_config = tune.TuneConfig(num_samples=samples, trial_dirname_creator=self.trial_dir_name,
										   max_concurrent_trials=3, search_alg=search,
										   scheduler=scheduler)

	def trial_dir_name(self, params) -> str:
		save_time = datetime.now().strftime('%Y%m%dT%H%M%S')
		param_s = f'{params}'
		hashed = f'{hash(param_s)}'
		self.dir_num += 1
		return f'{save_time}_{hashed}_{self.dir_num}'

	def tune_classifier_model(self, restore=True):

		ray.init(_temp_dir=(self.working_dir / 'out'))

		if restore and Tuner.can_restore(self.checkpoint_store):
			mytune = Tuner.restore(self.checkpoint_store, trainable=self.trainable_with_resources)
		else:
			mytune = Tuner(self.trainable_with_resources, param_space=self.search_space,
						   tune_config=self.tune_config)
		results_grid = mytune.fit()
		ray.shutdown()

		best_result: Result = results_grid.get_best_result(metric='v_loss', mode='min')
		checkpoint = best_result.checkpoint

		device = torch.device('cpu')

		with checkpoint.as_directory() as checkpoint_dir:
			classifier_params, model_state = torch.load(os.path.join(checkpoint_dir, "model_checkpoint.pth"),
														map_location=device)
			print(f'The dict is {classifier_params}')
			model = Classifier(**classifier_params)
			model.load_state_dict(model_state)

		config = best_result.config
		metrics: Result.metrics = best_result.metrics
		timestamp = metrics.timestamp

		checkpoint_name = f'classifier_best_{timestamp}.pth'

		AppLog.info(f'{checkpoint} with params: {model.model_params}')

		AppLog.info(f'Checkpoint: {checkpoint_name} saved with metric {metrics} used config {config}')

		torch.save((model.model_params, model.state_dict()), self.working_dir / 'models' / checkpoint_name)

		AppLog.shut_down()





def create_classifier_from_config(classifier_config) -> Classifier:
	classifier_config['final_size'] = int(round(classifier_config['final_size'], 0))
	classifier_config['dnn_layers'] = int(round(classifier_config['dnn_layers'], 0))
	classifier_config['cnn_layers'] = int(round(classifier_config['cnn_layers'], 0))
	classifier_config['starting_channels'] = int(classifier_config['starting_channels'])
	classifier_config['final_channels'] = int(classifier_config['final_channels'])
	layers = classifier_config['cnn_layers']
	dnn_layers = classifier_config['dnn_layers']
	channels = classifier_config['final_channels']
	size = classifier_config['final_size']
	starting_channels = classifier_config['starting_channels']

	flattened_shape_params: int = channels * size * size
	dnn_param_downscale_ratio = (10 / flattened_shape_params) ** (1 / dnn_layers)
	dnn_params = [int(round(flattened_shape_params * dnn_param_downscale_ratio ** dnn_l, 0)) for dnn_l in
				  range(1, dnn_layers + 1)]
	classifier = Classifier(dnn_params, 32, size, starting_channels, channels, layers)
	return classifier





def tune_with_exp(exp_model: ExperimentModels, config):
	result = exp_model.execute_single_experiment(config, config['batch_size'], config['learning_rate'])
	best_vloss = result['v_loss']
	trainable_params = result['trainable_params']
	model_params = result['model_params']

	AppLog.info(f'Best vloss: {best_vloss}, with {trainable_params} params. Performance per param (Higher is b'
				f'etter) = {1 / (trainable_params * best_vloss)}')
	AppLog.info(f'Classifier best vloss: {best_vloss}, training done. Model params: {model_params}.')

	return {'v_loss': best_vloss}


if __name__ == '__main__':
	# Shutdown ray if that is running
	ray.shutdown()
	experiment_time = datetime.now().strftime('%Y%m%dT%H%M%S')
	working_dir = Path.cwd()
	samples = 100
	tune_classifier = TuneClassifier(working_dir=working_dir, samples=samples)
	tune_classifier.tune_classifier_model()
	AppLog.shut_down()
