import os
import tempfile
from multiprocessing import Queue

import torch
from ray import tune
from ray.tune import Checkpoint
from torch import nn
from torchinfo import summary

from src.utils.common_utils import AppLog


class TrainModel:
    def __init__(self, save_checkpoint_epoch, model, loss_fn, optimizer, device, starting_epoch, ending_epoch) -> None:
        self.save_checkpoint_epoch = save_checkpoint_epoch
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.current_epoch = starting_epoch
        self.ending_epoch = ending_epoch
        self.best_vloss = float('inf')

    def train(self, train_loader):
        self.model.train(True)
        running_loss = 0.0
        train_batch_index = 0
        epoch = self.current_epoch
        EPOCHS = self.ending_epoch
        for img, label in train_loader:
            img, label = img.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            train_batch_index += 1

            raw_prob, prob = self.model.forward(img)
            loss = self.loss_fn(raw_prob, label)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            AppLog.info(
                f'Epoch [{epoch + 1}/{EPOCHS}]: Batch [{train_batch_index}]: Loss: {running_loss / train_batch_index}')
        avg_loss = running_loss / train_batch_index
        return avg_loss

    def evaluate(self, validation_loader):
        self.model.eval()
        running_vloss = 0.0
        valid_batch_index = 0
        epoch = self.current_epoch
        EPOCHS = self.ending_epoch
        with torch.no_grad():
            for img, label in validation_loader:
                img, label = img.to(self.device), label.to(self.device)
                raw_prob, prob = self.model.forward(img)
                v_loss = self.loss_fn(raw_prob, label)
                running_vloss += v_loss.item()
                valid_batch_index += 1
                AppLog.info(
                    f'Epoch [{epoch + 1}/{EPOCHS}]: V_Batch [{valid_batch_index}]: V_Loss: {running_vloss / (valid_batch_index)}')
        avg_vloss = running_vloss / valid_batch_index
        return avg_vloss

    def train_and_evaluate(self, train_loader, validation_loader):
        no_improvement = 0
        loss_best_threshold = 1.2

        while self.current_epoch < self.ending_epoch:

            avg_loss = self.train(train_loader)
            avg_vloss = self.evaluate(validation_loader)

            AppLog.info(f'Epoch {self.current_epoch + 1}: Training loss = {avg_loss}, Validation Loss = {avg_vloss}')

            self.save_checkpoint_epoch(avg_vloss, self.model, self.current_epoch)
            if avg_vloss < self.best_vloss:
                self.best_vloss = avg_vloss
            elif avg_vloss > loss_best_threshold * self.best_vloss:
                AppLog.warning(
                    f'Early stopping at {self.current_epoch + 1} epochs as (validation loss = {avg_vloss})/(best validation loss = {self.best_vloss}) > {loss_best_threshold} ')
                break
            elif no_improvement > 4:
                AppLog.warning(
                    f'Early stopping at {self.current_epoch + 1} epochs as validation loss = {avg_vloss} has shown no improvement over {no_improvement} epochs')
                break

            else:
                no_improvement += 1

            self.current_epoch += 1

        return self.best_vloss, self.model.model_params


# serialization_queue = Queue(maxsize=40)


class ExperimentModels:

    def __init__(self, model_creator_func, loader_func) -> None:
        self.model_creator_func = model_creator_func
        self.loader_func = loader_func

        # self.save_process = Process(target=self.queued_result, args=(serialization_queue,))  # self.save_process.start()

    def save_checkpoint(self, avg_vloss: float, classifier: nn.Module, epoch: int) -> None:
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            model_name_temp = f'classifier_tuned.pth'
            torch.save((classifier.model_params, classifier.state_dict()),
                       os.path.join(temp_checkpoint_dir, model_name_temp))
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            tune.report({'v_loss': avg_vloss, 'epoch': (epoch + 1)}, checkpoint=checkpoint)

    def queued_result(self, q: Queue) -> None:
        avg_vloss, classifier, epoch = q.get()
        self.save_checkpoint(avg_vloss, classifier, epoch)

    def execute_single_experiment(self, model_config, batch_size, lr):
        model = self.model_creator_func(model_config)
        train_loader, validation_loader = self.loader_func(batch_size)
        model_summary = summary(model, input_size=(batch_size, 3, 32, 32), verbose=0)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainable_params = model_summary.trainable_params
        AppLog.info(f'There are {trainable_params} trainable parameters.')

        train_model = TrainModel(self.save_checkpoint, model, loss_fn, optimizer, device, 0, 50)
        best_vloss, model_params = train_model.train_and_evaluate(train_loader, validation_loader)
        AppLog.info(
            f'Best vloss: {best_vloss}, with {trainable_params} params. Performance per param (Higher is better) = {1 / (trainable_params * best_vloss)}')
        AppLog.info(f'Classifier best vloss: {best_vloss}, training done. Model params: {model_params}.')
        return {'v_loss': best_vloss, 'trainable_params': trainable_params, 'model_params': model_params}

    # def shutdown_checkpoint(self):  #     self.save_process.close()  #     self.save_process.join()
