
import torch

from src.Model import Classifier
from src.common_utils import AppLog
from src.train_classifier import load_cifar_dataset


def show_images(img_b):
    img = img_b.numpy().transpose((1, 2, 0))
    un_normal = img*0.5+0.5



def load_model(model_name):
    classifier_params, model_state = torch.load(f'../models/{model_name}')
    saved_classifier = Classifier(**classifier_params)
    saved_classifier.load_state_dict(model_state)
    return saved_classifier


def get_state_and_show():
    saved_classifier = load_model('classifier_2')
    saved_classifier = saved_classifier.cuda()
    saved_classifier.eval()
    tr, valid = load_cifar_dataset(16)
    for img, label in tr:
        img = img.cuda()
        result, normed = saved_classifier.forward(img)
        max_value = torch.argmax(result,dim=1)
        AppLog.info(f'Evaluated = {max_value}, original = {label}')
        break

get_state_and_show()



