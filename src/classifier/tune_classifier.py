import os
from datetime import datetime
from pathlib import Path

import ray
import torch
from ray import tune
from ray.tune import Result, RunConfig, Tuner
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from src.classifier.classifier import Classifier
from src.classifier.training import ExperimentModels, load_cifar_dataset
from src.common.common_utils import AppLog


class TuneClassifier:
	def __init__(self, working_dir: Path, samples):
		self.dir_num = 0
		self.working_dir = working_dir
		self.checkpoint_store = self.working_dir / 'checkpoints'
		scheduler = ASHAScheduler(metric='v_loss', mode='min', time_attr='epoch', max_t=90, grace_period=12,
		                          reduction_factor=2, brackets=2)
		search = OptunaSearch(metric='v_loss', mode='min', study_name='tune_classifier')
		self.tune_run_config = RunConfig(name='tune_classifier_augmented', storage_path=f'{self.checkpoint_store}', )
		experiment = ExperimentModels(create_classifier_from_config,
									  lambda batch: load_cifar_dataset(self.working_dir, int(batch)))
		tune_exp = lambda tune_params: tune_with_exp(experiment, tune_params)

		self.search_space = {
				'fcn_layers'    : 3,
				'starting_channels': 54,
				'cnn_layers'    : 7,
				'final_channels': tune.quniform(256 - 32, 256 + 32, 2),
				'batch_size'    : tune.quniform(100, 500, 25),
				'lr'            : tune.loguniform(1e-5, 3e-4),
				'decay'         : tune.uniform(0, 1)
		}

		self.trainable_with_resources = tune.with_resources(tune_exp, {"cpu": 1, "gpu": 0.5})
		self.tune_config = tune.TuneConfig(num_samples=samples, trial_dirname_creator=self.trial_dir_name,
										   max_concurrent_trials=2, scheduler=scheduler, search_alg=search)

	# This is done as sometimes ray tune creates directories with invalid characters
	def trial_dir_name(self, params) -> str:
		save_time = datetime.now().strftime('%Y%m%dT%H%M%S')
		param_s = f'{params}'
		hashed = f'{hash(param_s)}'
		self.dir_num += 1
		return f'a_{save_time}_{hashed}_{self.dir_num}'


	def tune_classifier_model(self, restore=True):

		tempdir = self.working_dir / 'out'
		ray.init(_temp_dir=f'{tempdir.absolute()}')
		location = f'{(self.checkpoint_store / "tune_classifier").absolute()}'
		AppLog.info(f'Restoring from: {location}')
		if restore and Tuner.can_restore(self.checkpoint_store):
			mytune = Tuner.restore(f'{location}', trainable=self.trainable_with_resources)
		else:
			mytune = Tuner(self.trainable_with_resources, param_space=self.search_space, tune_config=self.tune_config,
						   run_config=self.tune_run_config)
		results_grid = mytune.fit()
		ray.shutdown()

		best_result: Result = results_grid.get_best_result(metric='v_loss', mode='min')
		checkpoint = best_result.checkpoint

		device = torch.device('cpu')

		# This part may be omitted as the checkpoint directory can be used to acquire the best version of the model
		# later.
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

	channels, fcn_layers, layers, starting_channels = prepare_classifier_params(classifier_config)
	final_size = 2
	flattened_shape_params: int = channels * final_size * final_size
	fcn_param_downscale_ratio = (10 / flattened_shape_params) ** (1 / fcn_layers)
	fcn_params = [int(round(flattened_shape_params * fcn_param_downscale_ratio ** fcn_l, 0)) for fcn_l in
				  range(1, fcn_layers + 1)]
	classifier = Classifier(fcn_params, 32, final_size, starting_channels, channels, layers)
	return classifier


def prepare_classifier_params(classifier_config):
	classifier_config['fcn_layers'] = int(round(classifier_config['fcn_layers'], 0))
	classifier_config['cnn_layers'] = int(round(classifier_config['cnn_layers'], 0))
	classifier_config['starting_channels'] = int(classifier_config['starting_channels'])
	cnn_layers = classifier_config['cnn_layers']
	fcn_layers = classifier_config['fcn_layers']
	channels = int(round(classifier_config['final_channels']))
	starting_channels = int(round(classifier_config['starting_channels'], 0))
	return channels, fcn_layers, cnn_layers, starting_channels


def tune_with_exp(exp_model: ExperimentModels, config):
	result = exp_model.execute_single_experiment(config, config['batch_size'], config['lr'])
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
	# Ensure that the working dir is the base of the project.
	working_dir = Path.cwd()
	torch.set_float32_matmul_precision('high')
	samples = 150
	tune_classifier = TuneClassifier(working_dir=working_dir, samples=samples)
	tune_classifier.tune_classifier_model()
	AppLog.shut_down()
