# all the metrics are averaged over each batch in default.
import warnings

import torch
import torch.nn as nn
from torch.nn import functional as F


class Accuracy:
    """Calculate accuracy of a model"""
    def __init__(self):
        self.__name__ = "accuracy"
        self._base = [0, 0]

    def calc(self, outputs, targets, reduction='mean'):
        """ Compute the accuracy.
        Args:
        ------
        outputs: torch.Tensor
        The output of the model, shape (batch_size, num_classes)

        targets: torch.Tensor
        The ground truth label, shape (batch_size, )

        reduction: str
        The reduction method, 'mean' or 'sum'
        If 'mean', return the mean accuracy of the batch
        If 'sum', return the sum of correct predictions of the batch

        Returns:
        --------
        accuracy: torch.Tensor
        """
        _, preds = torch.max(outputs, 1)
        correct = torch.sum(preds == targets)

        if reduction == 'mean':
            return correct / len(targets)
        elif reduction == 'sum':
            return correct
        else:
            raise ValueError("reduction must be 'mean' or 'sum'")

    def accumulate(self, outputs, targets):
        """ Accumulate the metric over batches."""
        _, preds = torch.max(outputs, 1)
        correct = torch.sum(preds == targets)
        self.base[0] += correct
        self.base[1] += len(targets)

    def reset(self):
        self._base = [0, 0]

    def accumulated_score(self):
        """ Return the accumulated score in one epoch."""
        if self._base[1] == 0:
            # divide by zero warning
            warnings.warn("The denominator is zero, return 0", RuntimeWarning)
            return 0
        return self._base[0] / self.base[1]

    def __call__(self, outputs, targets, reduction='mean'):
        return self.calc(outputs, targets, reduction)

class Precision:
    """Calculate precision of a model"""
    def __init__(self):
        self.__name__ = "precision"
        self._base = [0, 0]

    def calc(self, outputs, targets, reduction='mean'):
        """ Compute the precision.
        Args:
        ------
        outputs: torch.Tensor
        The output of the model, shape (batch_size, num_classes)

        targets: torch.Tensor
        The ground truth label, shape (batch_size, )

        reduction: str
        The reduction method, 'mean' or 'sum'
        If 'mean', return the mean precision of the batch
        If 'sum', return the sum of correct predictions of the batch

        Returns:
        --------
        precision: torch.Tensor
        """
        _, preds = torch.max(outputs, 1)
        correct = torch.sum(preds == targets)
        total = torch.sum(preds == preds)

        if reduction == 'mean':
            return correct / total
        elif reduction == 'sum':
            return correct
        else:
            raise ValueError("reduction must be 'mean' or 'sum'")

    def accumulate(self, outputs, targets):
        """ Accumulate the metric over batches."""
        _, preds = torch.max(outputs, 1)
        correct = torch.sum(preds == targets)
        total = torch.sum(preds == preds)
        self.base[0] += correct
        self.base[1] += total

    def reset(self):
        self._base = [0, 0]

    def accumulated_score(self):
        """ Return the accumulated score in one epoch."""
        if self._base[1] == 0:
            # divide by zero warning
            warnings.warn("The denominator is zero, return 0", RuntimeWarning)
            return 0
        return self._base[0] / self.base[1]

    def __call__(self, outputs, targets, reduction='mean'):
        return self.calc(outputs, targets, reduction)


class MSE:
    """Calculate mean squared error of a model"""
    def __init__(self):
        self.__name__ = "mse"
        self._base = [0, 0]

    def calc(self, outputs, targets, reduction='mean'):
        """
        Args:
        ------
        outputs: torch.Tensor
        The output of the model, shape (batch_size,)

        targets: torch.Tensor
        The ground truth label, shape (batch_size,)

        reduction: str
        The reduction method, 'mean' or 'sum'

        Returns:
        --------
        mse: torch.Tensor
        """
        if reduction == 'mean':
            return F.mse_loss(outputs, targets, reduction='mean')
        elif reduction == 'sum':
            return F.mse_loss(outputs, targets, reduction='sum')
        else:
            raise ValueError("reduction must be 'mean' or 'sum'")

    def accumulate(self, outputs, targets):
        """ Accumulate the metric over batches."""
        self.base[0] += F.mse_loss(outputs, targets, reduction='sum')
        self.base[1] += len(targets)

    def reset(self):
        self._base = [0, 0]

    def accumulated_score(self):
        """ Return the accumulated score in one epoch."""
        if self._base[1] == 0:
            # divide by zero warning
            warnings.warn("The denominator is zero, return 0", RuntimeWarning)
            return 0
        return self._base[0] / self.base[1]

    def __call__(self, outputs, targets, reduction='mean'):
        return self.calc(outputs, targets, reduction)


class MAE:
    """Calculate mean absolute error of a model"""
    def __init__(self):
        self.__name__ = "mae"
        self._base = [0, 0]

    def calc(self, outputs, targets, reduction='mean'):
        """
        Args:
        ------
        outputs: torch.Tensor
        The output of the model, shape (batch_size,)

        targets: torch.Tensor
        The ground truth label, shape (batch_size,)

        reduction: str
        The reduction method, 'mean' or 'sum'

        Returns:
        --------
        mae: torch.Tensor
        """
        if reduction == 'mean':
            return F.l1_loss(outputs, targets, reduction='mean')
        elif reduction == 'sum':
            return F.l1_loss(outputs, targets, reduction='sum')
        else:
            raise ValueError("reduction must be 'mean' or 'sum'")

    def accumulate(self, outputs, targets):
        """ Accumulate the metric over batches."""
        self.base[0] += F.l1_loss(outputs, targets, reduction='sum')
        self.base[1] += len(targets)

    def reset(self):
        self._base = [0, 0]

    def accumulated_score(self):
        """ Return the accumulated score in one epoch."""
        if self._base[1] == 0:
            # divide by zero warning
            warnings.warn("The denominator is zero, return 0", RuntimeWarning)
            return 0
        return self._base[0] / self.base[1]

    def __call__(self, outputs, targets, reduction='mean'):
        return self.calc(outputs, targets, reduction)






