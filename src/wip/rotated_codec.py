from multiprocessing import Process
from queue import Queue

import torch
import tempfile
from ray import tune
from ray.tune import Checkpoint
import os

from torchinfo import summary

from src.utils.common_utils import AppLog


class TrainModel:
    def __init__(self, save_pooling, model, loss_fn, optimizer, device, starting_epoch, ending_epoch):
        self.save_pooling = save_pooling
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
            # img, label = img.to(self.device), label.to(self.device)
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
                # img, label = img.to(device), label.to(device)
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

            self.save_pooling.send((avg_vloss, self.model, self.current_epoch,))
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

        return self.best_vloss,  self.model.model_params


class ExperimentModels:

    @classmethod
    def save_checkpoint(cls,avg_vloss, classifier, epoch):
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            model_name_temp = f'classifier_tuned.pth'
            torch.save(
                (classifier.model_params, classifier.state_dict()),
                os.path.join(temp_checkpoint_dir, model_name_temp)
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            tune.report({'v_loss': avg_vloss, 'epoch': (epoch + 1)}, checkpoint=checkpoint)

    @classmethod
    def queued_result(cls,q : Queue):
        avg_vloss, classifier, epoch = q.get()
        cls.save_checkpoint(avg_vloss, classifier, epoch)



    def __init__(self,model_creator_func,loader_func):

        self.model_creator_func = model_creator_func
        self.loader_func = loader_func
        self.serialization_queue = Queue(maxsize=40)
        self.save_process = Process(target=self.queued_result, args=(self.serialization_queue,))
        self.save_process.start()



    def send_checkpoint(self,check_params):
        self.serialization_queue.put(check_params)


    def execute_single_experiment(self, model_config, batch_size, lr):
        model = self.model_creator_func(model_config)
        train_loader, validation_loader = self.loader_func(batch_size)
        model_summary = summary(model, input_size=(batch_size, 3, 32, 32), verbose=0)
        AppLog.info(f'{model_summary}')


        train_model = TrainModel(self.send_checkpoint,model, train_loader, validation_loader)












