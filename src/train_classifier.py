from datetime import datetime

import torch
from torch import nn
from torchinfo import summary
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from src.Model import Classifier
from src.common_utils import AppLog


def load_cifar_dataset(batch=512):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    training_set = CIFAR10(root='../data/CIFAR/train', train=True, download=True, transform=transform)
    validation_set = CIFAR10(root='../data/CIFAR/test', train=False, download=True, transform=transform)
    AppLog.info(f'{len(training_set)} training samples and {len(validation_set)} validation samples')
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch, shuffle=True)
    return train_loader, validation_loader


def train_one_epoch(epoch, EPOCHS, train_loader, validation_loader, model, optimizer, loss_fn, device):
    model.train(True)
    running_loss = 0.0
    train_batch_index = 0
    for img, label in train_loader:
        optimizer.zero_grad()
        img = img.device(device)
        label = label.device(device)
        train_batch_index += 1

        raw_prob, prob = model.forward(img)
        loss = loss_fn(raw_prob, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        AppLog.info(
            f'Epoch [{epoch + 1}/{EPOCHS}]: Batch [{train_batch_index}]: Loss: {running_loss / train_batch_index}')
    avg_loss = running_loss / train_batch_index

    # Evaluate the classifier.
    model.eval()
    running_vloss = 0.0
    valid_batch_index = 0
    with torch.no_grad():
        for img, label in validation_loader:
            img, label = img.device(device), label.device(device)
            raw_prob, prob = model.forward(img)
            v_loss = loss_fn(raw_prob, label)
            running_vloss += v_loss.item()
            valid_batch_index += 1
            AppLog.info(
                f'Epoch [{epoch + 1}/{EPOCHS}]: V_Batch [{valid_batch_index}]: V_Loss: {running_vloss / (valid_batch_index)}')

    avg_vloss = running_vloss / valid_batch_index

    AppLog.info(f'Epoch {epoch + 1}: Training loss = {avg_loss}, Validation Loss = {avg_vloss}')

    return avg_vloss, avg_loss


def tune_classifier(learning_rate, dnn_layers, final_size, starting_channels, final_channels, cnn_layers, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, validation_loader = load_cifar_dataset(batch_size)
    flattened_shape_params = final_channels * final_size * final_size
    dnn_param_downscale_ratio = (10 / flattened_shape_params) ** (1 / dnn_layers)
    dnn_params = [round(flattened_shape_params * dnn_param_downscale_ratio ** dnn_l) for dnn_l in
                  range(1, dnn_layers + 1)]

    classifier = Classifier(dnn_params, 32, final_size, starting_channels, final_channels, cnn_layers).device(device)
    AppLog.info(f'{summary(classifier, input_size=(batch_size, 3, 32, 32))}')
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    EPOCHS = 40
    best_vloss = float('inf')
    AppLog.info(f'Starting {EPOCHS} epochs with learning rate {learning_rate}')

    best_model_name = ''
    for epoch in range(EPOCHS):
        avg_vloss, avg_loss = train_one_epoch(epoch, EPOCHS, train_loader, validation_loader, classifier, optimizer,
                                              loss_fn, device)

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            save_time = datetime.now().strftime('%Y%m%dT%H%M%S')
            model_name = f'classifier_{save_time}_{epoch + 1}.pth'
            best_model_name = model_name
            torch.save((classifier.model_params, classifier.state_dict()),
                       f'../models/{model_name}')
    return best_vloss, best_model_name, classifier.model_params


if __name__ == '__main__':
    best_vloss, best_model_name, model_params = tune_classifier(0.001, 4, 2, 32, 192, 6, 500)

    AppLog.info(f'Classifier best vloss: {best_vloss}, training done. Model params: {model_params}')
