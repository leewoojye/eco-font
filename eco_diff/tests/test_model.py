import torch

from eco_diff.models import EcoMaskUNet


def test_model_shape():
    model = EcoMaskUNet(input_channels=6, base_channels=8, depth=3)
    x = torch.randn(2, 6, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)
