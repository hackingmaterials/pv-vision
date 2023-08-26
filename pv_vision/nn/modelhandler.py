# Author: Xin Chen w/Copilot
# Date: 2023-03-27
# Description: ModelHandler class
# Note this is not tested with unit tests yet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import random_split
from tqdm import tqdm
import pickle


class ModelHandler:
    def __init__(self,
                 model,
                 model_output=None,
                 train_dataset=None,
                 val_dataset=None,
                 test_dataset=None,
                 predict_only = False,
                 batch_size_train=128,
                 batch_size_val=128,
                 learning_rate=0.001,
                 lr_scheduler=None,
                 num_epochs=15,
                 criterion=None,
                 optimizer=None,
                 device=None,
                 evaluate_metric=None,
                 log_interval=10,
                 save_dir='checkpoints',
                 save_name='model.pt'
                 ) -> None:
        """ The ModelHandler class is used to train, validate, and test a model.

        Args:
        ----------
        model: nn.Module
        Model to train

        model_output: String
        Some models from Pytorch output a dictionary that contains the predcition.
        The key should be given explicitly.
        Default is None.

        train_dataset: torch.utils.data.Dataset
        Dataset to train the model

        val_dataset: torch.utils.data.Dataset
        Dataset to validate the model

        test_dataset: torch.utils.data.Dataset
        Dataset to test the model

        predict_only: bool
        If True, only make prediction without training. Default is False.

        batch_size_train/val: int
        Batch size to use for training. Default is 128. Batch_size_val is only used if predict_only is True.

        learning_rate: float
        Learning rate to use for training. Default is 0.001.

        lr_scheduler: torch.optim.lr_scheduler
        Learning rate scheduler to use for training. Default is None.

        num_epochs: int
        Number of epochs to train the model. Default is 15.

        criterion: torch.nn.modules.loss
        Loss function to use for training. Default is None.

        optimizer: torch.optim.Optimizer
        Optimizer to use for training. Default is None.

        evaluate_metric: Object
        Metric class to evaluate the model. Default is None and only loss is reported.

        device: torch.device
        Device to use for training. Default is None.

        log_interval: int
        Number of batches to wait before logging training status. Default is 10.

        save_dir: str
        Directory to save the model. Default is 'checkpoints'.

        save_name: str
        Name of the model to save. Default is 'model.pt'.

        Returns:
        ----------
        None
        """

        self.model = model
        self.model_output = model_output

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.predict_only = predict_only
        if self.predict_only:
            if self.train_dataset is not None or self.val_dataset is not None:
                raise ValueError('train_dataset and val_dataset must be None when predict_only is True')
            if self.test_dataset is None:
                raise ValueError('test_dataset must be defined when predict_only is True')

        self.batch_size_val = batch_size_val
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size_val, shuffle=False)

        if self.predict_only is False:
            self.batch_size_train = batch_size_train

            self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size_train, shuffle=True)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size_val, shuffle=False)

            if learning_rate is None:
                raise ValueError('learning_rate is not defined')
            else:
                self.learning_rate = learning_rate

            self.lr_scheduler = lr_scheduler
            self.num_epochs = num_epochs
            if criterion is None:
                raise ValueError('criterion is not defined')
            else:
                self.criterion = criterion

            # check if evaluate_metric is a class
            # check if evaluate_metric has __name__ attribute
            if evaluate_metric is not None:
                if not isinstance(evaluate_metric, type):
                    raise ValueError('evaluate_metric is not a class')
                if not hasattr(evaluate_metric, '__name__'):
                    raise ValueError('evaluate_metric does not have __name__ attribute')
                self.evaluate_metric = evaluate_metric()
            else:
                self.evaluate_metric = None


            self.optimizer = optimizer if optimizer is not None else optim.Adam(self.model.parameters())

            # One dictionary that record loss and metric of training and validation for each epoch
            # use the name of the metric as the key. If no metric is defined, only use 'loss' as the key
            self.running_record = {'train': {'loss': []}, 'val': {'loss': []}}
            if self.evaluate_metric is not None:
                self.running_record['train'][self.evaluate_metric.__name__] = []
                self.running_record['val'][self.evaluate_metric.__name__] = []

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_name = save_name

        self.log_interval = log_interval
        self._setup_logging()
        self.cache = {}

    def _setup_logging(self):
        """ Setup logging """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # create a file handler
        handler = logging.FileHandler(os.path.join(self.save_dir, 'modelhandler.log'))
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(handler)

        # initial message
        self.logger.info('ModelHandler initialized.')

    def _save_model(self, epoch):
        """ Save model """
        os.makedirs(os.path.join(self.save_dir, f'epoch_{epoch}'), exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'epoch_{epoch}', self.save_name))
        self.logger.info(f'Saved model at epoch {epoch}')

    def _train(self, epoch):
        """ Helper function to train model in one epoch
        Return loss value and metric value if evaluate_metric is defined.
        """
        self.model.train()
        loss_epoch = 0.0
        metric_epoch = None if self.evaluate_metric is None else 0.0
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader)):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            if self.model_output:
                output = output[self.model_output]
            
            # check if criterion is cross entropy loss becasue the output shape is different
            if isinstance(self.criterion, nn.CrossEntropyLoss):
                loss = self.criterion(output, target.long())
            else:
                loss = self.criterion(output, target.reshape(output.shape).float())
            loss_epoch += loss.item() * data.size(0)
            if self.evaluate_metric is not None:
                metric = self.evaluate_metric(output, target.reshape(output.shape).float())
                metric_epoch += metric

            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                info1 = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item())
                info2 = '' if self.evaluate_metric is None else f'\t{self.evaluate_metric.__name__}: {metric}'
                self.logger.info(info1+info2)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self._save_model(epoch)

        loss_epoch /= len(self.train_loader.dataset)
        if self.evaluate_metric is not None:
            metric_epoch /= len(self.train_loader.dataset)

        return loss_epoch, metric_epoch

    def _evaluate(self, dataloader, cache_output=False):
        """ Evaluate model for val/test.
        Return the loss. If evaluate_metric is defined, also return the metric value.
        If cache_output is given, will cache the output of model. The value of cache_output
        will be the key of cache.
        """
        # cache_output should be either False or String
        if (not isinstance(cache_output, str)) and (not isinstance(cache_output, bool)):
                raise TypeError('cache_output must be either False or String')
        self.model.eval()
        loss = 0.0
        metric = None if self.evaluate_metric is None else 0.0
        if cache_output:
            outall = []
        with torch.no_grad():
            for data, target in tqdm(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                if self.model_output:
                    output = output[self.model_output]
                if cache_output:
                    outall.append(output.cpu().numpy())
                # check if criterion is cross entropy loss becasue the output shape is different
                if isinstance(self.criterion, nn.CrossEntropyLoss):
                    loss += self.criterion(output, target.long()).item() * data.size(0)
                else:
                    loss += self.criterion(output, target.reshape(output.shape).float()).item() * data.size(0)
                if self.evaluate_metric is not None:
                    metric += self.evaluate_metric(output, target.reshape(output.shape).float()).item()
        if cache_output:
            self.cache[cache_output] = np.concatenate(outall)
        loss /= len(dataloader.dataset)
        if self.evaluate_metric is not None:
            metric /= len(dataloader.dataset)
        return loss, metric

    def _validate(self, epoch):
        """ Helper function to validate model in one epoch
        Return loss value and metric value if evaluate_metric is defined.
        Log the loss and metric value.
        """
        loss, metric = self._evaluate(self.val_loader)
        self.logger.info('Val epoch: {} \tAverage loss: {:.4f}'.format(epoch, loss))
        if self.evaluate_metric is not None:
            self.logger.info(f'{self.evaluate_metric.__name__}: {metric}')
        return loss, metric

    def test_model(self, cache_output=False):
        """ Test model on test set """
        print('Testing mode')
        print('-' * 10)
        if self.predict_only:
            raise ValueError('Cannot test model when predict_only is True')
        loss, metric = self._evaluate(self.test_loader, cache_output)

        self.logger.info(f'Test set: Average loss: {loss:.4f}')
        if self.evaluate_metric is not None:
            self.logger.info(f'{self.evaluate_metric.__name__}: {metric}')
        
        print(f'Test set: Average loss: {loss:.4f}')
        if self.evaluate_metric is not None:
            print(f'{self.evaluate_metric.__name__}: {metric}')
        return loss, metric

    def train_model(self, print_memory=False):
        """ Train and evaluate model """
        if self.predict_only:
            raise ValueError('Cannot train model when predict_only is True')
        for epoch in range(1, self.num_epochs + 1):
            print(f'Epoch {epoch} / {self.num_epochs}')
            print('-' * 10)

            loss_train, metric_train = self._train(epoch)
            loss_val, metric_val = self._validate(epoch)

            print(f'Loss: {loss_train:.4f} (train) | {loss_val:.4f} (val)')
            if self.evaluate_metric is not None:
                print(f'{self.evaluate_metric.__name__}: {metric_train:.4f} (train) | {metric_val:.4f} (val)')

            if print_memory:
                # if not use cuda, this will not work
                if self.device.type == 'cuda':
                    total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**2 # in MB
                    allocated_memory = torch.cuda.memory_allocated(self.device) / 1024**2 # in MB
                    reserved_memory = torch.cuda.memory_reserved(self.device) / 1024**2 # in MB
                    free_memory = total_memory - reserved_memory
                    print(f'Total memory: {total_memory:.2f} MB')
                    print(f'Allocated memory: {allocated_memory:.2f} MB')
                    print(f'Reserved memory: {reserved_memory:.2f} MB')
                    print(f'Free memory: {free_memory:.2f} MB')
                else:
                    print('Not using GPU')

            self.running_record['train']['loss'].append(loss_train)
            self.running_record['val']['loss'].append(loss_val)
            if self.evaluate_metric is not None:
                self.running_record['train'][self.evaluate_metric.__name__].append(metric_train)
                self.running_record['val'][self.evaluate_metric.__name__].append(metric_val)
        with open(os.path.join(self.save_dir, 'running_record.pkl'), 'wb') as f:
            pickle.dump(self.running_record, f)
        return self.running_record

    def load_model(self, path):
        """ Load model from path """
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        self.logger.info(f'Loaded model from {path}')

    def predict(self, save=False):
        """ Predict on new dataloader that doesn't have labels
        If save, save all the predictions in one matrix 
        """
        self.model.eval()
        output = []
        with torch.inference_mode():
            for data in tqdm(self.test_loader):
                data = data.to(self.device)
                out = self.model(data)
                if self.model_output:
                    out = out[self.model_output]
                output.append(out.cpu().numpy())
        output = np.concatenate(output)
        if save:
            np.save(os.path.join(self.save_dir, self.save_name), output)
            self.logger.info(f'Saved prediction to {os.path.join(self.save_dir, self.save_name)}')
            
        return output

    def predict_proba(self):
        """ Predict probability on dataloader that doesn't have labels"""
        pass
        # self.model.eval()
        # output = []
        # with torch.inference_mode():
        #     for data in self.test_loader:
        #         data = data.to(self.device)
        #         output.append(torch.sigmoid(self.model(data)).cpu().numpy())
        # return np.concatenate(output)

    def plot_loss(self):
        """ Plot loss curve with seaborn """
        sns.lineplot(data=self.running_record['train']['loss'], label='train')
        sns.lineplot(data=self.running_record['val']['loss'], label='val')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    def plot_metric(self):
        """ Plot metric curve """
        if self.predict_only:
            raise ValueError('Cannot plot metric when predict_only is True')
        sns.lineplot(data=self.running_record['train'][self.evaluate_metric.__name__], label='train')
        sns.lineplot(data=self.running_record['val'][self.evaluate_metric.__name__], label='val')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel(self.evaluate_metric.__name__)
        plt.show()








