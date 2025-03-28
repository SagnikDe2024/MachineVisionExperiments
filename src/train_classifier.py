import asyncio
import os
import tempfile
from datetime import datetime

import numpy as np
import ray
import torch
from ray import tune
from ray.tune import Checkpoint
from ray.tune import Result
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
# from ray.train.torch import prepare_model
# from ray.train.v2.torch.train_loop_utils import prepare_data_loader
from torch import nn
from torchinfo import summary
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from src.models.Model import Classifier
from src.utils.common_utils import AppLog
from src.wip.training import ExperimentModels


# import ray.train.torch


def load_cifar_dataset(batch: int=500):
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


def train_one_epoch(epoch: int, EPOCHS: int, train_loader, validation_loader, model: Classifier, optimizer: torch.optim.adam.Adam, loss_fn: torch.nn.modules.loss.CrossEntropyLoss, device):
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


def tune_classifier(learning_rate, classifier: Classifier, batch_size, device: torch.device):
    train_loader, validation_loader = load_cifar_dataset(batch_size)

    model_summary = summary(classifier, input_size=(batch_size, 3, 32, 32), verbose=0)
    trainable_params = model_summary.trainable_params

    AppLog.info(f'{model_summary}')
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    train_time = datetime.now().strftime('%Y%m%dT%H%M%S')
    param_string = f'{classifier.model_params}'
    model_name = f'classifier_{train_time}_{hash(param_string)}.pth'

    classifier = classifier.to(device)

    EPOCHS = 50
    no_improvement = 0
    loss_best_threshold = 1.2
    best_vloss = float('inf')
    AppLog.info(f'Starting {EPOCHS} epochs with learning rate {learning_rate}')
    # loop = asyncio.get_event_loop()
    # checkpoint_tasks = set()
    best_model_name = ''
    for epoch in range(EPOCHS):
        # if ray.train.get_context().get_world_size() > 1:
        #     train_loader.sampler.set_epoch(epoch)
        #     validation_loader.sampler.set_epoch(epoch)
        avg_vloss, avg_loss = train_one_epoch(epoch, EPOCHS, train_loader, validation_loader, classifier, optimizer,
                                              loss_fn, device)

        asyncio.run(save_checkpoint(avg_vloss, classifier, epoch))
        # checkpoint_tasks.add(check_point_create)

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            AppLog.info(f'Not saving model at {epoch + 1} epoch, best vloss {best_vloss}')

            # best_model_name = model_name  # torch.save((classifier.model_params, classifier.state_dict()),  #            f'C:/mywork/python/ImageEncoderDecoder/models/{model_name}')
        elif avg_vloss > loss_best_threshold * best_vloss:
            AppLog.warning(
                f'Early stopping at {epoch + 1} epochs as (validation loss = {avg_vloss})/(best validation loss = {best_vloss}) > {loss_best_threshold} ')
            break
        elif no_improvement > 4:
            AppLog.warning(
                f'Early stopping at {epoch + 1} epochs as validation loss = {avg_vloss} has shown no improvement over {no_improvement} epochs')
            break
        else:
            no_improvement += 1
    # checkpoint_tasks

    return best_vloss, best_model_name, classifier.model_params, trainable_params


async def save_checkpoint(avg_vloss: float, classifier: Classifier, epoch: int) -> None:
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        model_name_temp = f'classifier_tuned.pth'
        torch.save((classifier.model_params, classifier.state_dict()),
            os.path.join(temp_checkpoint_dir, model_name_temp))
        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
        tune.report({'v_loss': avg_vloss, 'epoch': (epoch + 1)}, checkpoint=checkpoint)


def tune_classifier_aux(config):
    classifier = create_classifier_from_config(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_vloss, best_model_name, model_params, trainable_params = tune_classifier(config['learning_rate'], classifier,
                                                                                  config['batch_size'], device)
    AppLog.info(
        f'Best vloss: {best_vloss}, with {trainable_params} params. Performance per param (Higher is better) = {1 / (trainable_params * best_vloss)}')
    AppLog.info(
        f'Classifier best vloss: {best_vloss}, training done. Model params: {model_params}. Saved model: {best_model_name} ')

    return {'v_loss': best_vloss}


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

    AppLog.info(
        f'Best vloss: {best_vloss}, with {trainable_params} params. Performance per param (Higher is better) = {1 / (trainable_params * best_vloss)}')
    AppLog.info(f'Classifier best vloss: {best_vloss}, training done. Model params: {model_params}.')

    return {'v_loss': best_vloss}


if __name__ == '__main__':
    ray.shutdown()
    ray.init(local_mode=True, _temp_dir='C:/mywork/python/ImageEncoderDecoder/out')

    scheduler = ASHAScheduler(metric='v_loss', mode='min', time_attr='epoch', max_t=40, grace_period=5,
                              reduction_factor=3)

    optuna_search = OptunaSearch(metric='v_loss', mode='min')

    search_space = {'learning_rate'    : tune.loguniform(0.0001, 0.01),
                    'dnn_layers'       : tune.quniform(4, 7, 1),
                    'final_size'       : tune.quniform(1, 4, 1),
                    'starting_channels': tune.qloguniform(12, 32, 4),
                    'final_channels'   : tune.sample_from(lambda spec : np.random.uniform(256/(spec.config.final_size**2),2048/(spec.config.final_size**2))),
                    'cnn_layers'       : tune.quniform(3, 6, 1),
                    'batch_size'       : tune.choice([125,250, 500, 1000, 2000])}

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
