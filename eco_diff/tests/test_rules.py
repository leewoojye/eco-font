import numpy as np

from eco_diff.metrics import apply_cut_mask, evaluate_candidate
from eco_diff.rules import select_best_candidate


def test_rule_candidate_saves_some_ink():
    glyph = np.zeros((64, 64), dtype=np.float32)
    glyph[12:52, 18:46] = 1.0
    candidate = select_best_candidate(glyph, 0.25)
    eco = apply_cut_mask(glyph, candidate.cut_mask)
    metrics = evaluate_candidate(glyph, candidate.cut_mask)
    assert eco.sum() < glyph.sum()
    assert metrics.ink_saving >= 0.0
    assert metrics.ssim <= 1.0
