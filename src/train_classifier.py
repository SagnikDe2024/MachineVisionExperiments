from datetime import datetime

import ray
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
# from ray.train.torch import prepare_model
# from ray.train.v2.torch.train_loop_utils import prepare_data_loader
from torch import nn
from torchinfo import summary
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from src.models.Model import Classifier
from src.utils.common_utils import AppLog


# import ray.train.torch


def load_cifar_dataset(batch=512):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    training_set = CIFAR10(root='C:/mywork/python/ImageEncoderDecoder/data/CIFAR/train', train=True, download=True,
                           transform=transform)
    validation_set = CIFAR10(root='C:/mywork/python/ImageEncoderDecoder/data/CIFAR/test', train=False, download=True,
                             transform=transform)
    AppLog.info(f'{len(training_set)} training samples and {len(validation_set)} validation samples')
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch, shuffle=True)
    return train_loader, validation_loader


def train_one_epoch(epoch, EPOCHS, train_loader, validation_loader, model, optimizer, loss_fn, device):
    model.train(True)
    running_loss = 0.0
    train_batch_index = 0
    for img, label in train_loader:
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
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
            img, label = img.to(device), label.to(device)
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

    # train_loader = prepare_data_loader(train_loader)
    # validation_loader = prepare_data_loader(validation_loader)

    flattened_shape_params = final_channels * final_size * final_size
    dnn_param_downscale_ratio = (10 / flattened_shape_params) ** (1 / dnn_layers)
    dnn_params = [int(round(flattened_shape_params * dnn_param_downscale_ratio ** dnn_l, 0)) for dnn_l in
                  range(1, dnn_layers + 1)]

    classifier = Classifier(dnn_params, 32, final_size, starting_channels, final_channels, cnn_layers).to(device)
    # classifier = ray.train.torch.prepare_model(classifier)

    model_summary = summary(classifier, input_size=(batch_size, 3, 32, 32))
    trainable_params = model_summary.trainable_params

    AppLog.info(f'{model_summary}')
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    train_time = datetime.now().strftime('%Y%m%dT%H%M%S')
    param_string = f'{classifier.model_params}'
    model_name = f'classifier_{train_time}_{hash(param_string)}.pth'

    EPOCHS = 40
    no_improvement = 0
    loss_best_threshold = 1.2
    best_vloss = float('inf')
    AppLog.info(f'Starting {EPOCHS} epochs with learning rate {learning_rate}')

    best_model_name = ''
    for epoch in range(EPOCHS):
        # if ray.train.get_context().get_world_size() > 1:
        #     train_loader.sampler.set_epoch(epoch)
        #     validation_loader.sampler.set_epoch(epoch)
        avg_vloss, avg_loss = train_one_epoch(epoch, EPOCHS, train_loader, validation_loader, classifier, optimizer,
                                              loss_fn, device)
        # loss_metrics = {'avg_vloss': avg_vloss, 'avg_loss': avg_loss}
        # with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        #     save_time = datetime.now().strftime('%Y%m%dT%H%M%S')
        #     model_name = f'classifier_{save_time}_{epoch + 1}.pth'
        #     torch.save(
        #         (classifier.model_params, classifier.state_dict()),
        #         os.path.join(temp_checkpoint_dir, model_name)
        #     )
        #     ray.train.report(loss_metrics, checkpoint_path=temp_checkpoint_dir)
        # if ray.train.get_context().get_world_rank() == 0:
        #     AppLog.info(f'Metrics in an epoch {loss_metrics}')

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss

            best_model_name = model_name
            # torch.save((classifier.model_params, classifier.state_dict()), f'../models/{model_name}')
            torch.save((classifier.model_params, classifier.state_dict()),
                       f'C:/mywork/python/ImageEncoderDecoder/models/{model_name}')
        elif avg_vloss > loss_best_threshold * best_vloss:
            AppLog.warning(
                f'Early stopping at {epoch + 1} epochs as (validation loss = {avg_vloss})/(best validation loss = {best_vloss}) > {loss_best_threshold} ')
            break
        elif no_improvement > 5:
            AppLog.warning(
                f'Early stopping at {epoch + 1} epochs as validation loss = {avg_vloss} has shown no improvement over {no_improvement} epochs')
            break
        else:
            no_improvement += 1

    return best_vloss, best_model_name, classifier.model_params, trainable_params


def tune_classifier_aux(config):
    config['final_size'] = int(round(config['final_size'], 0))
    config['dnn_layers'] = int(round(config['dnn_layers'], 0))
    config['cnn_layers'] = int(round(config['cnn_layers'], 0))
    config['starting_channels'] = int(config['starting_channels'])
    config['final_channels'] = int(config['final_channels'])

    best_vloss, best_model_name, model_params, trainable_params = tune_classifier(**config)
    AppLog.info(
        f'Best vloss: {best_vloss}, with {trainable_params} params. Performance per param (Higher is better) = {1 / (trainable_params * best_vloss)}')
    AppLog.info(
        f'Classifier best vloss: {best_vloss}, training done. Model params: {model_params}. Saved model: {best_model_name} ')

    return {'v_loss': best_vloss}


store = [0]


def trail_dir_name(params):
    save_time = datetime.now().strftime('%Y%m%dT%H%M%S')
    param_s = f'{params}'
    hashed = f'{hash(param_s)}'
    store[0] += 1
    return f'{save_time}_{hashed}_{store[0]}'


if __name__ == '__main__':
    # scaling_config = ScalingConfig(use_gpu=True)
    # run_config = RunConfig(storage_path='../models', name='tuned_classifier.pth')
    ray.init(local_mode=True, _temp_dir='C:/mywork/python/ImageEncoderDecoder/out')
    trainable_with_resources = tune.with_resources(tune_classifier_aux, {"gpu": 0.9})

    search_space = {'learning_rate': tune.loguniform(0.01, 0.0001), 'dnn_layers': tune.quniform(3, 6, 1),
                    'final_size': tune.quniform(1, 4, 1), 'starting_channels': tune.qloguniform(12, 32, 4),
                    'final_channels': tune.qloguniform(128, 384, 64), 'cnn_layers': tune.quniform(3, 6, 1),
                    'batch_size': tune.grid_search([500, 1000, 2000])}
    tuner = tune.Tuner(trainable_with_resources, param_space=search_space,
                       tune_config=tune.TuneConfig(num_samples=3, trial_dirname_creator=trail_dir_name,
                                                   scheduler=ASHAScheduler(metric='v_loss', mode='min')))
    results = tuner.fit()
    AppLog.info(f"{results.get_best_result('v_loss')}")

    # best_vloss, best_model_name, model_params = tune_classifier(0.001, 4, 2, 32, 192, 6, 500)

    # AppLog.info(f'Classifier best vloss: {best_vloss}, training done. Model params: {model_params}. Saved model: {best_model_name} ')
