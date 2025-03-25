import torch

from src.Model import Classifier
from src.common_utils import AppLog
from src.train_classifier import load_cifar_dataset

from torcheval.metrics.functional import multiclass_confusion_matrix


def show_images(img_b):
    img = img_b.numpy().transpose((1, 2, 0))
    un_normal = img * 0.5 + 0.5


def load_model(model_name):
    classifier_params, model_state = torch.load(f'../models/{model_name}')
    saved_classifier = Classifier(**classifier_params)
    saved_classifier.load_state_dict(model_state)
    return saved_classifier


def get_state_and_show():
    saved_classifier = load_model('classifier_20250325T091707_17.pth')
    saved_classifier = saved_classifier.cuda()
    saved_classifier.eval()
    tr, valid = load_cifar_dataset(1000)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

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


get_state_and_show()
