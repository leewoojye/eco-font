# Inference Pipeline Summary

This document summarizes the current generation model and inference pipeline
used in `eco_research_hangul`.

## Core Idea

The current system does not rely on a single model output as the final font.
It uses a trained conditional diffusion model as one candidate generator, then
selects the final glyph with OCR, ink saving, style, and design constraints.

In short:

```text
source font glyph
  -> candidate generation
  -> OCR / ink / style / design scoring
  -> final eco glyph
```

## Generation Model

The learned model is a small conditional DDPM.

- Model file: `src/eco_research_hangul/model.py`
- Diffusion schedule: `src/eco_research_hangul/diffusion.py`
- Training entry: `src/eco_research_hangul/train.py`
- Checkpoint: `runs/smoke/diffusion_best.pt`

Training data was made from paired rendered glyph images:

- source fonts: normal Nanum Hangul fonts
- target fonts: real Nanum Eco fonts
- target style: perforated eco font targets

Because this supervised target is Nanum Eco, raw diffusion samples often learn
dot/hole patterns. For that reason, diffusion output is now treated as a
candidate, not the final answer.

## Candidate Generation

For each input character, the pipeline renders the source glyph and creates
multiple candidates.

Candidate groups:

- `source_original`
- `source_erode1`, `source_erode2`
- `source_inline_*`
- `source_erode*_inline_*`
- `diffusion_*`
- `diffusion_*_closed_*`
- diffusion candidates with erode / inline variants

Candidate generation code:

- `src/eco_research_hangul/guided.py`
- preview helper: `src/eco_research_hangul/candidate_preview.py`

## Guided Selection

The final glyph is selected by objective scoring.

Main criteria:

- Tesseract OCR exact match
- ink saving against the source glyph
- Ryman Eco style reference via VGG Gram-style score
- design gate against small holes
- design gate against small foreground fragments
- penalty for excessive ink removal

The final selected candidate must pass OCR and design gates when possible.
This is why recent results mostly avoid the repeated dot-hole pattern.

## OCR

OCR uses the system Tesseract binary.

For Hangul:

```yaml
ocr_lang: kor
ocr_psm: [8, 6]
```

For Cherokee:

```yaml
ocr_lang: chr
ocr_psm: [8, 6]
```

Multi-PSM OCR is used because some fonts are recognized better under different
Tesseract page segmentation modes.

## Current Experiment Configs

Hangul examples:

- `configs/guided.yaml`
- `configs/guided_myeongjo.yaml`
- `configs/guided_barunpen.yaml`
- `configs/guided_jua.yaml`

Cherokee example:

- `configs/guided_cherokee.yaml`

## Main Commands

Run guided inference:

```bash
../.venv/bin/eco-research-hangul guided-infer --config configs/guided_jua.yaml
../.venv/bin/eco-research-hangul report --root outputs/guided_jua --output outputs/guided_jua/contact_sheet.png
```

Generate candidate intermediate outputs:

```bash
../.venv/bin/eco-research-hangul candidate-preview --config configs/guided_jua.yaml --output-dir outputs/candidate_preview_jua
../.venv/bin/eco-research-hangul candidate-preview --config configs/guided_cherokee.yaml --output-dir outputs/candidate_preview_cherokee
```

## Practical Interpretation

The current pipeline is best understood as:

```text
optimization / reranking based eco glyph generation
```

not as:

```text
pure end-to-end diffusion font generation
```

Diffusion contributes diversity, but OCR and design gates often prefer
source-based erode or inline candidates because they preserve readability and
avoid perforated dot artifacts.

