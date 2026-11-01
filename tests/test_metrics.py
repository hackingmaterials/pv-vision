"""Tests for the per-batch metrics in pv_vision.nn.metrics."""
import pytest
import torch

from pv_vision.nn.metrics import MAE, MSE, Accuracy


def test_accuracy_mean_and_sum():
    outputs = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    targets = torch.tensor([0, 1, 1, 1])
    acc = Accuracy()
    assert acc(outputs, targets).item() == pytest.approx(0.75)
    assert acc(outputs, targets, reduction="sum").item() == 3


def test_accuracy_rejects_unknown_reduction():
    with pytest.raises(ValueError):
        Accuracy()(torch.zeros(2, 2), torch.zeros(2, dtype=torch.long), reduction="max")


def test_mse_and_mae_on_known_values():
    outputs = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.0, 4.0, 0.0])
    assert MSE()(outputs, targets).item() == pytest.approx((0 + 4 + 9) / 3)
    assert MAE()(outputs, targets).item() == pytest.approx((0 + 2 + 3) / 3)
