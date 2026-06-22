import numpy as np
import torch

from eco_diff.diffusion import DiffusionSchedule
from eco_diff.diffusion_model import ConditionalDiffusionUNet
from eco_diff.evaluator import enforce_ink_budget, evaluate_eco_candidate


def test_diffusion_model_shape():
    model = ConditionalDiffusionUNet(condition_channels=6, base_channels=8, depth=3)
    noisy = torch.randn(2, 1, 64, 64)
    cond = torch.randn(2, 6, 64, 64)
    t = torch.tensor([0, 5], dtype=torch.long)
    out = model(noisy, cond, t)
    assert out.shape == noisy.shape


def test_schedule_q_sample_shape():
    schedule = DiffusionSchedule(timesteps=8)
    x0 = torch.randn(2, 1, 32, 32)
    t = torch.tensor([0, 7], dtype=torch.long)
    noisy = schedule.q_sample(x0, t)
    assert noisy.shape == x0.shape


def test_enforce_ink_budget_reduces_area():
    original = np.zeros((32, 32), dtype=np.float32)
    original[6:26, 8:24] = 1.0
    candidate = original.copy()
    projected = enforce_ink_budget(candidate, original, 0.5)
    metrics = evaluate_eco_candidate(original, projected, 0.5)
    assert projected.sum() < original.sum()
    assert abs(metrics.ink_saving - 0.5) < 0.15


def test_template_ocr_matches_simple_shape():
    original = np.zeros((32, 32), dtype=np.float32)
    original[6:26, 8:24] = 1.0
    other = np.zeros((32, 32), dtype=np.float32)
    other[8:24, 6:26] = np.eye(16, 20, dtype=np.float32)
    metrics = evaluate_eco_candidate(
        original,
        original,
        0.0,
        expected_char="A",
        template_ocr={"A": original, "B": other},
    )
    assert metrics.template_ocr_text == "A"
    assert metrics.template_ocr_match is True
