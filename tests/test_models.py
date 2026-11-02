"""Shape tests for the network architectures in pv_vision.nn.models.

Released trained weights depend on these architectures, so a change that
alters layer shapes should fail here before it reaches users.
"""
import torch

from pv_vision.nn.models import LeNet


def test_lenet_outputs_five_classes():
    model = LeNet().eval()
    with torch.no_grad():
        out = model(torch.rand(4, 3, 32, 32))
    assert out.shape == (4, 5)
    assert torch.isfinite(out).all()


def test_lenet_layer_shapes_are_stable():
    shapes = {name: tuple(p.shape) for name, p in LeNet().named_parameters()}
    assert shapes == {
        "conv1.weight": (6, 3, 5, 5), "conv1.bias": (6,),
        "conv2.weight": (16, 6, 5, 5), "conv2.bias": (16,),
        "fc1.weight": (120, 400), "fc1.bias": (120,),
        "fc2.weight": (84, 120), "fc2.bias": (84,),
        "fc3.weight": (5, 84), "fc3.bias": (5,),
    }
