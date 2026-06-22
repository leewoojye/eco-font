import numpy as np
import torch

from eco_research_hangul.dataset import condition_from_source
from eco_research_hangul.diffusion import DiffusionSchedule
from eco_research_hangul.metrics import ink_saving
from eco_research_hangul.model import build_model


def test_condition_shape():
    source = np.zeros((32, 32), dtype=np.float32)
    cond = condition_from_source(source, 0.3)
    assert cond.shape == (4, 32, 32)


def test_model_shape():
    model = build_model({"condition_channels": 4, "base_channels": 8, "depth": 2})
    noisy = torch.randn(2, 1, 32, 32)
    cond = torch.randn(2, 4, 32, 32)
    t = torch.randint(0, 8, (2,))
    out = model(noisy, cond, t)
    assert out.shape == noisy.shape


def test_diffusion_q_sample_shape():
    schedule = DiffusionSchedule(timesteps=8)
    x0 = torch.randn(2, 1, 32, 32)
    t = torch.randint(0, 8, (2,))
    out = schedule.q_sample(x0, t)
    assert out.shape == x0.shape


def test_ink_saving():
    src = np.ones((8, 8), dtype=np.float32)
    eco = np.zeros((8, 8), dtype=np.float32)
    assert ink_saving(src, eco) == 1.0

