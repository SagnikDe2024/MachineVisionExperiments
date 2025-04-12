import json
from pathlib import Path

import pandas as pd
import torch
from pandas import read_csv
from torchinfo import summary

from src.classifier.classifier import Classifier
from src.tune_classifier import load_cifar_dataset, prepare_classifier_params
from src.utils.common_utils import AppLog


def show_images(img_b) -> None:
	img = img_b.numpy().transpose((1, 2, 0))
	un_normal = img * 0.5 + 0.5

def visTensor(tensor, nrows = 8, padding=1) -> None:
	n,c,h,w = tensor.shape
	tensor = tensor.view(n,c,h,w).cpu().detach().numpy()



def remove_empty_checkpoint_directory():
	working_dir = Path.cwd()
	classifier_checkpoints_dir = working_dir / 'checkpoints' / 'tune_classifier'
	sub_directories = [sub for sub in classifier_checkpoints_dir.iterdir() if sub.is_dir()]
	for sub_directory in sub_directories:
		stored_checkpoint_dirs = [chk for chk in sub_directory.iterdir() if chk.is_dir()]

		if len(stored_checkpoint_dirs) == 0:
			print(f'No checkpoint directory found for {sub_directory}')
			stored_checkpoint_files = [chk for chk in sub_directory.iterdir() if chk.is_file()]
			for file in stored_checkpoint_files:
				file.unlink()
			sub_directory.rmdir()
			continue


def find_classify_checkpoint():
	working_dir = Path.cwd()
	classifier_checkpoints_dir = working_dir / 'checkpoints' / 'tune_classifier'
	sub_directories = [sub for sub in classifier_checkpoints_dir.iterdir() if sub.is_dir()]
	param_dict = {'batch_size'   : [], 'cnn_layers': [], 'fcn_layers': [], 'final_channels': [],
				  'learning_rate': [], 'starting_channels': [], 'v_loss': [], 'total_params': [], 'checkpoint_path':
					  []}
	for sub_directory in sub_directories:
		params_json_path = sub_directory / 'params.json'
		progress_csv_path = sub_directory / 'progress.csv'
		if not (params_json_path.exists() and progress_csv_path.exists()):
			continue

		tune_params = json.load(open(params_json_path))

		progress = read_csv(progress_csv_path)
		min_vloss: float = progress.v_loss.min()

		checkpoint_dir_min_vloss_index = progress[
			progress.v_loss == min_vloss].index[0]
		checkpoint_dir_min_vloss = progress.at[checkpoint_dir_min_vloss_index, 'checkpoint_dir_name']

		AppLog.info(f'Tuning {sub_directory} has vloss {min_vloss}')
		AppLog.info(checkpoint_dir_min_vloss)
		AppLog.info(f'Parameters: {tune_params}')
		checkpoint = sub_directory / checkpoint_dir_min_vloss / 'model_checkpoint.pth'
		classifier_params, model_state = torch.load(checkpoint)
		classifier = Classifier(**classifier_params)
		batch_size = tune_params['batch_size']
		model_with_params = summary(classifier, (100, 3, 32, 32), verbose=0)
		total_params = model_with_params.trainable_params

		final_channels, fcn_layers, cnn_layers, starting_channels = prepare_classifier_params(tune_params)

		param_dict['batch_size'].append(batch_size)
		param_dict['cnn_layers'].append(cnn_layers)
		param_dict['fcn_layers'].append(fcn_layers)
		param_dict['final_channels'].append(final_channels)
		param_dict['learning_rate'].append(tune_params['learning_rate'])
		param_dict['starting_channels'].append(starting_channels)
		param_dict['total_params'].append(total_params)
		param_dict['v_loss'].append(min_vloss)
		param_dict['checkpoint_path'].append(checkpoint)

	v_loss_df = pd.DataFrame(param_dict)
	return v_loss_df


def get_state_and_show_accuracy(checkpoint_path) -> dict[str, float]:
	classifier_params, model_state = torch.load(checkpoint_path)
	classifier = Classifier(**classifier_params)
	# WTF ?! The compiled model is saved with a `_orig_mod` in the keys. Apparently the weights are shared with
	# uncompiled model but there is no documentation for that ...

	for key in list(model_state.keys()):
		model_state[key.replace("_orig_mod.", "")] = model_state.pop(key)
	classifier.load_state_dict(model_state)
	best_classifier = classifier.cuda()
	summary(best_classifier, (100, 3, 32, 32))
	best_classifier = torch.compile(best_classifier)
	working_dir = Path.cwd()

	best_classifier.eval()
	tr, valid = load_cifar_dataset(working_dir, 1000)
	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	correct_pred = {classname: 0 for classname in classes}
	total_pred = {classname: 0 for classname in classes}
	AppLog.info(f'Showing perf of {checkpoint_path}')

	with torch.no_grad():
		for img, labels in valid:
			img = img.cuda()
			result, normed = best_classifier.forward(img)
			pred_labels = torch.argmax(normed, dim=1).cpu()

			# conf_matrix = multiclass_confusion_matrix(pred_labels, labels, 10)

			for label, prediction in zip(labels, pred_labels):
				if label == prediction:
					correct_pred[classes[label]] += 1
				total_pred[classes[label]] += 1

	class_accuracy: dict[str, float] = {}
	for classname, correct_count in correct_pred.items():
		accuracy = 100 * float(correct_count) / total_pred[classname]
		class_accuracy[classname] = accuracy
		AppLog.info(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
	return class_accuracy

def show_model_accuracy() -> None:
	get_state_and_show_accuracy('classifier_best.pth')


if __name__ == '__main__':
	# result_df = find_classify_checkpoint()
	# min_vloss = result_df.v_loss.min()
	# min_vloss_row = result_df[
	# 	result_df.v_loss == min_vloss]
	# pd.options.display.max_columns = None
	# print(f'{min_vloss_row}')
	# checkpoint_used = min_vloss_row.to_dict()['checkpoint_path']
	#
	# check = [v for k,v in checkpoint_used.items()][0]
	# AppLog.info(f'Checkpoint used: {check}')
	work_path = Path.cwd()
	chkpath = (work_path / 'checkpoints' / 'tune_classifier' / '4_20250408T035811_-8491173734139600649_1' /
			   'checkpoint_000009' / 'model_checkpoint.pth')
	# path = Path('C:\mywork\python\MachineVisionExperiments\checkpoints\tune_classifier\4_20250408T035811_
	# -8491173734139600649_1\checkpoint_000009\model_checkpoint.pth')
	get_state_and_show_accuracy(chkpath)

	AppLog.shut_down()

# removed_empty_checkpoint_directory()
# show_model_accuracy()
