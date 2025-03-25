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


if __name__ == '__main__':

    train_loader, validation_loader = load_cifar_dataset(500)

    classifier = Classifier([384, 114, 34, 10], 32, 2, 16, 256, 6).cuda()
    AppLog.info(f'{summary(classifier, input_size=(500, 3, 32, 32))}')
    learning_rate = 0.001
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    EPOCHS = 10
    best_vloss = float('inf')
    AppLog.info(f'Starting {EPOCHS} epochs with learning rate {learning_rate}')

    for epoch in range(EPOCHS):
        classifier.train(True)
        running_loss = 0.0
        train_batch_index = 0
        for img, label in train_loader:
            img = img.cuda()
            label = label.cuda()
            train_batch_index += 1

            raw_prob, prob = classifier.forward(img)
            loss = loss_fn(raw_prob, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            AppLog.info(
                f'Epoch [{epoch + 1}/{EPOCHS}]: Batch [{train_batch_index}]: Loss: {running_loss / train_batch_index}')

        # Evaluate the classifier.
        classifier.eval()
        running_vloss = 0.0
        valid_batch_index = 0
        with torch.no_grad():
            for img, label in validation_loader:
                img, label = img.cuda(), label.cuda()
                img = img.cuda()
                raw_prob, prob = classifier.forward(img)
                v_loss = loss_fn(raw_prob, label)
                running_vloss += v_loss.item()
                valid_batch_index += 1
                AppLog.info(
                    f'Epoch [{epoch + 1}/{EPOCHS}]: V_Batch [{valid_batch_index}]: V_Loss: {running_vloss / (valid_batch_index)}')

        avg_loss = running_loss / train_batch_index
        avg_vloss = running_vloss / valid_batch_index
        AppLog.info(f'Epoch {epoch + 1}: Training loss = {avg_loss}, Validation Loss = {avg_vloss}')

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            torch.save((classifier.model_params, classifier.state_dict()), f'../models/classifier_{epoch + 1}')

    AppLog.info(f'Classifier best vloss: {best_vloss}, training done')
