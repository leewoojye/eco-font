import numpy as np
import torch

from ryman_font.model import RymanNet
from ryman_font.pseudo import input_channels, make_ryman_target
from ryman_font.metrics import evaluate, ink_saving, recognize_by_tesseract


def test_target_saves_ink():
    glyph = np.zeros((64, 64), dtype=np.float32)
    glyph[10:54, 18:46] = 1.0
    target, score = make_ryman_target(glyph, 0.45)
    assert target.shape == glyph.shape
    assert score.shape == glyph.shape
    assert ink_saving(glyph, target) > 0.25


def test_distinct_target_saves_ink():
    glyph = np.zeros((64, 64), dtype=np.float32)
    glyph[10:54, 18:46] = 1.0
    target, score = make_ryman_target(glyph, 0.62, style="distinct")
    assert target.shape == glyph.shape
    assert score.shape == glyph.shape
    assert ink_saving(glyph, target) > 0.45


def test_canonical_target_saves_ink():
    glyph = np.zeros((64, 64), dtype=np.float32)
    glyph[10:54, 18:46] = 1.0
    target, score = make_ryman_target(glyph, 0.68, style="canonical", char="한")
    assert target.shape == glyph.shape
    assert score.shape == glyph.shape
    assert ink_saving(glyph, target) > 0.55


def test_cherokee_canonical_target_uses_canvas_prior():
    glyph = np.zeros((64, 64), dtype=np.float32)
    glyph[12:52, 18:46] = 1.0
    target, score = make_ryman_target(glyph, 0.62, style="canonical", char="Ꭰ")
    assert target.shape == glyph.shape
    assert score.shape == glyph.shape
    assert ink_saving(glyph, target) > 0.45
    assert score.max() > 0.0


def test_model_shape():
    model = RymanNet(input_channels=7, base_channels=8, depth=3)
    x = torch.randn(2, 7, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)


def test_input_channels_shape():
    glyph = np.zeros((32, 32), dtype=np.float32)
    glyph[8:24, 8:24] = 1.0
    x = input_channels(glyph, 0.4)
    assert x.shape == (7, 32, 32)


def test_distinct_input_channels_shape():
    glyph = np.zeros((32, 32), dtype=np.float32)
    glyph[8:24, 8:24] = 1.0
    x = input_channels(glyph, 0.62, style="distinct")
    assert x.shape == (7, 32, 32)


def test_canonical_input_channels_shape():
    glyph = np.zeros((32, 32), dtype=np.float32)
    glyph[8:24, 8:24] = 1.0
    x = input_channels(glyph, 0.68, style="canonical", char="한")
    assert x.shape == (7, 32, 32)


def test_tesseract_result_schema_when_unavailable_or_available():
    glyph = np.zeros((32, 32), dtype=np.float32)
    glyph[8:24, 8:24] = 1.0
    result = recognize_by_tesseract(glyph, expected_char="한", lang="kor")
    assert {"available", "text", "confidence", "match", "error"} <= set(result)


def test_evaluate_without_template_does_not_count_template_match():
    glyph = np.zeros((32, 32), dtype=np.float32)
    glyph[8:24, 8:24] = 1.0
    metrics = evaluate(glyph, glyph, 0.0, expected_char="한", ocr_engine="none")
    assert metrics["template_ocr_match"] is None
    assert metrics["tesseract_ocr_match"] is None
