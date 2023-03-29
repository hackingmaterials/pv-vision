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

class ModelHandler:
    def __init__(self,
                 model,
                 train_dataset,
                 val_dataset,
                 test_dataset,
                 batch_size=128,
                 learning_rate=0.001,
                 lr_scheduler=None,
                 num_epochs=15,
                 criterion=None,
                 optimizer=None,
                 device=None,
                 evaluate_metric=None,
                 save_dir='checkpoints',
                 save_name='model.pt'
                 ) -> None:
        """ The ModelHandler class is used to train, validate, and test a model.

        Args:
        ----------
        model: nn.Module
        Model to train

        train_dataset: torch.utils.data.Dataset
        Dataset to train the model

        val_dataset: torch.utils.data.Dataset
        Dataset to validate the model

        test_dataset: torch.utils.data.Dataset
        Dataset to test the model

        batch_size: int
        Batch size to use for training. Default is 128.

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

        save_dir: str
        Directory to save the model. Default is 'checkpoints'.

        save_name: str
        Name of the model to save. Default is 'model.pt'.

        Returns:
        ----------
        None
        """

        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

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

        self.evaluate_metric = evaluate_metric
        # check if evaluate_metric has __name__ attribute
        if self.evaluate_metric is not None:
            if not hasattr(self.evaluate_metric, '__name__'):
                raise ValueError('evaluate_metric must have __name__ attribute')

        self.optimizer = optimizer if optimizer is not None else optim.Adam(self.model.parameters())
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # One dictionary that record loss and metric of training and validation for each epoch
        # use the name of the metric as the key. If no metric is defined, only use 'loss' as the key
        self.running_record = {'train': {'loss':[]}, 'val': {'loss':[]}}
        if self.evaluate_metric is not None:
            self.running_record['train'][self.evaluate_metric.__name__] = []
            self.running_record['val'][self.evaluate_metric.__name__] = []

        self.log_interval = 10
        self._setup_logging()

        self.save_dir = save_dir
        os.mkdir(self.save_dir, exist_ok=True)
        self.save_name = save_name

    def _setup_logging(self):
        """ Setup logging """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # create a file handler
        handler = logging.FileHandler(os.path.join(self.save_dir, 'training.log'))
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(handler)

    def _save_model(self, epoch):
        """ Save model """
        os.mkdir(os.path.join(self.save_dir, f'epoch_{epoch}'), exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'epoch_{epoch}', self.save_name))
        self.logger.info(f'Saved model at epoch {epoch}')

    def _train(self, epoch):
        """ Helper function to train model in one epoch
        Return loss value and metric value if evaluate_metric is defined.
        """
        self.model.train()
        loss_epoch = 0.0
        metric_epoch = None if self.evaluate_metric is None else 0.0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss_epoch += loss.item()
            if self.evaluate_metric is not None:
                metric = self.evaluate_metric(output, target)
                metric_epoch += metric

            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))

            if self.evaluate_metric is not None:
                self.logger.info(f'{self.evaluate_metric.__name__}: {metric}')

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self._save_model(epoch)

        loss_epoch /= len(self.train_loader.dataset)
        if self.evaluate_metric is not None:
            metric_epoch /= len(self.train_loader.dataset)

        return loss_epoch, metric_epoch

    def _evaluate(self, dataloader):
        """ Evaluate model for val/test.
        Return the loss. If evaluate_metric is defined, also return the metric value.
        """
        self.model.eval()
        loss = 0.0
        metric = None if self.evaluate_metric is None else 0.0
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += self.criterion(output, target).item()
                if self.evaluate_metric is not None:
                    metric += self.evaluate_metric(output, target).item()

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

    def test(self):
        """ Test model on test set """
        loss, metric = self._evaluate(self.test_loader)
        self.logger.info(f'Test set: Average loss: {loss:.4f}')
        if self.evaluate_metric is not None:
            self.logger.info(f'{self.evaluate_metric.__name__}: {metric}')

        return loss, metric

    def train(self):
        """ Train model """
        for epoch in range(1, self.num_epochs + 1):
            loss_train, metric_train = self._train(epoch)
            loss_val, metric_val = self._validate(epoch)

            self.running_record['train']['loss'].append(loss_train)
            self.running_record['val']['loss'].append(loss_val)
            if self.evaluate_metric is not None:
                self.running_record['train'][self.evaluate_metric.__name__].append(metric_train)
                self.running_record['val'][self.evaluate_metric.__name__].append(metric_val)

        return self.running_record

    def load_model(self, path):
        """ Load model from path """
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

    def predict(self, dataloader):
        """ Predict on dataloader """
        self.model.eval()
        output = []
        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                output.append(self.model(data).cpu().numpy())
        return np.concatenate(output)

    def predict_proba(self, dataloader):
        """ Predict probability on dataloader """
        pass
        # self.model.eval()
        # output = []
        # with torch.no_grad():
        #     for data in dataloader:
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
        sns.lineplot(data=self.running_record['train'][self.evaluate_metric.__name__], label='train')
        sns.lineplot(data=self.running_record['val'][self.evaluate_metric.__name__], label='val')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel(self.evaluate_metric.__name__)
        plt.show()








