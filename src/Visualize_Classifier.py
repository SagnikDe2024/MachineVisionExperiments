import torch
from torchinfo import summary

from src.models.Model import Classifier
from src.train_classifier import load_cifar_dataset
from src.utils.common_utils import AppLog


def show_images(img_b) -> None:
	img = img_b.numpy().transpose((1, 2, 0))
	un_normal = img * 0.5 + 0.5

def visTensor(tensor, nrows = 8, padding=1) -> None:
	n,c,h,w = tensor.shape
	tensor = tensor.view(n,c,h,w).cpu().detach().numpy()


def load_model(model_name) -> Classifier:
	classifier_params, model_state = torch.load(f'c:/mywork/python/ImageEncoderDecoder/models/{model_name}')
	saved_classifier = Classifier(**classifier_params)
	saved_classifier.load_state_dict(model_state)
	return saved_classifier


def get_state_and_show(file_name) -> None:
	saved_classifier = load_model(file_name)
	saved_classifier = saved_classifier.cuda()
	summary(saved_classifier, (1000, 3, 32, 32))
	saved_classifier = torch.compile(saved_classifier)

	saved_classifier.eval()
	tr, valid = load_cifar_dataset(1000)
	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	correct_pred = {classname: 0 for classname in classes}
	total_pred = {classname: 0 for classname in classes}
	AppLog.info(f'Showing perf of {file_name}')

	with torch.no_grad():
		for img, labels in tr:
			img = img.cuda()
			result, normed = saved_classifier.forward(img)
			pred_labels = torch.argmax(normed, dim=1).cpu()

			# conf_matrix = multiclass_confusion_matrix(pred_labels, labels, 10)

			for label, prediction in zip(labels, pred_labels):
				if label == prediction:
					correct_pred[classes[label]] += 1
				total_pred[classes[label]] += 1

	for classname, correct_count in correct_pred.items():
		accuracy = 100 * float(correct_count) / total_pred[classname]
		AppLog.info(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


def show_model_accuracy() -> None:
	get_state_and_show('classifier_best.pth')

	AppLog.shut_down()


if __name__ == '__main__':
	show_model_accuracy()
