import os
import tempfile
from datetime import datetime

import numpy as np
import ray
import torch
from ray import tune
from ray.tune import Checkpoint, Result
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from src.models.classifier import Classifier
from src.utils.common_utils import AppLog
from src.wip.training import ExperimentModels


# import ray.train.torch


def load_cifar_dataset(batch: int = 500):
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

	training_set = CIFAR10(root='C:/mywork/python/ImageEncoderDecoder/data/CIFAR/train', train=True, download=True,
						   transform=transform)
	validation_set = CIFAR10(root='C:/mywork/python/ImageEncoderDecoder/data/CIFAR/test', train=False, download=True,
							 transform=transform)
	AppLog.info(f'{len(training_set)} training samples and {len(validation_set)} validation samples')
	train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch, shuffle=True, pin_memory=True,
											   drop_last=True)
	validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=5000, shuffle=True, pin_memory=True,
													drop_last=True)

	return train_loader, validation_loader


async def save_checkpoint(avg_vloss: float, classifier: Classifier, epoch: int) -> None:
	with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
		model_name_temp = f'classifier_tuned.pth'
		torch.save((classifier.model_params, classifier.state_dict()),
				   os.path.join(temp_checkpoint_dir, model_name_temp))
		checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
		tune.report({'v_loss': avg_vloss, 'epoch': (epoch + 1)}, checkpoint=checkpoint)


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


store = [0]


def trail_dir_name(params) -> str:
	save_time = datetime.now().strftime('%Y%m%dT%H%M%S')
	param_s = f'{params}'
	hashed = f'{hash(param_s)}'
	store[0] += 1
	return f'{save_time}_{hashed}_{store[0]}'


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
	ray.shutdown()
	ray.init(local_mode=True, _temp_dir='C:/mywork/python/ImageEncoderDecoder/out')

	scheduler = ASHAScheduler(metric='v_loss', mode='min', time_attr='epoch', max_t=40, grace_period=5,
							  reduction_factor=3)

	optuna_search = OptunaSearch(metric='v_loss', mode='min')

	search_space = {'learning_rate'    : tune.loguniform(0.00001, 0.0075),
					'dnn_layers'       : tune.quniform(4, 7, 1),
					'final_size'       : tune.quniform(1, 4, 1),
					'starting_channels': tune.qloguniform(12, 32, 4),
					'final_channels'   : tune.sample_from(
						lambda spec: np.random.uniform(256 / (spec.config.final_size ** 2),
													   2048 / (spec.config.final_size ** 2))),
					'cnn_layers'       : tune.quniform(3, 6, 1),
					'batch_size'       : tune.choice([100,125, 250, 500, 1000])}

	experiment = ExperimentModels(create_classifier_from_config, load_cifar_dataset)
	tune_exp = lambda tune_params: tune_with_exp(experiment, tune_params)

	trainable_with_resources = tune.with_resources(tune_exp, {'cpu': 1, "gpu": 0.45})

	# config = tune.TuneConfig()
	# tune.Tuner.can_restore()

	tuner = tune.Tuner(trainable_with_resources, param_space=search_space,
					   tune_config=tune.TuneConfig(num_samples=100, trial_dirname_creator=trail_dir_name,
												   max_concurrent_trials=3, search_alg=optuna_search,
												   scheduler=scheduler))
	results_grid = tuner.fit()
	# experiment.shutdown_checkpoint()
	ray.shutdown()

	best_result: Result = results_grid.get_best_result(metric='v_loss', mode='min')
	checkpoint = best_result.checkpoint

	best_config = best_result.config

	model = None
	device = torch.device('cpu')

	with checkpoint.as_directory() as checkpoint_dir:
		classifier_params, model_state = torch.load(os.path.join(checkpoint_dir, "classifier_tuned.pth"),
													map_location=device)
		print(f'The dict is {classifier_params}')
		model = Classifier(**classifier_params)
		model.load_state_dict(model_state)

	config = best_result.config
	metrics = best_result.metrics

	checkpoint_name = f'classifier_best.pth'

	AppLog.info(f'{checkpoint} with params: {model.model_params}')
	# AppLog.info(f'model = {model.state_dict()}')
	AppLog.info(f'Checkpoint: {checkpoint_name} saved with metric {metrics} used config {config}')

	torch.save((model.model_params, model.state_dict()),
			   f'C:/mywork/python/ImageEncoderDecoder/models/{checkpoint_name}')

	AppLog.shut_down()
