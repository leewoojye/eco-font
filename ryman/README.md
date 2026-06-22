# ryman

Ryman-Eco-inspired experimental font generator.

This is a self-contained project under `eco-font/ryman`. It does not depend on
the sibling `eco_diff` package.

The goal is not to preserve the source font shape. The goal is to generate a new
eco typeface direction inspired by Ryman Eco:

- thin multi-line / contour-line strokes
- clear ink savings
- readable glyph structure
- visually intentional, not random holes
- optional TTF export from generated bitmap contours

## Pipeline

```text
TTF fonts -> rendered glyphs -> Ryman-style pseudo targets
          -> RymanNet U-Net training
          -> inference with ink-budget projection
          -> PNG previews + metrics + optional TTF
```

## Install

```bash
cd /home/woojye2020/decs_jupyter_lab/eco-font/ryman
PYTHONPATH=src python -m ryman_font.cli --help
```

Dependencies are listed in `requirements.txt`.

## Full Hangul Experiment

```bash
bash scripts/run_hangul_experiment.sh
```

This script:

1. Builds a pseudo-labeled Hangul dataset from several Nanum fonts.
2. Trains `RymanNet`.
3. Runs inference at 0.35, 0.45, and 0.55 target savings.
4. Exports a 0.45 TTF.
5. Builds a contact sheet and metrics summary.

## Style Variants

The project now includes three target styles:

- `contour`: closest to the first Ryman-like experiment. It preserves the source
  glyph skeleton and is best for OCR/readability stability.
- `distinct`: reduces outline preservation and emphasizes interior centerline
  rhythm. This is the current balanced candidate for high ink saving while still
  reading close to the source content.
- `canonical`: redraws Hangul from decomposed jamo with a modular line design.
  This is much less source-font-like, but it is more experimental and should be
  judged visually before treating OCR metrics as final.

All training configs set `skeleton_weight: 0.0`, so source-font skeleton
similarity is tracked as an analysis metric only. It is not part of the training
objective.

Run the stronger variants:

```bash
bash scripts/run_hangul_distinct_experiment.sh
bash scripts/run_hangul_canonical_experiment.sh
```

## OCR Evaluation

Inference uses Tesseract OCR by default:

```bash
PYTHONPATH=src python -m ryman_font.cli infer ... \
  --ocr-engine tesseract \
  --ocr-lang kor \
  --ocr-psm 10
```

This project calls the `tesseract` executable directly. For Hangul OCR, the
system needs both Tesseract and Korean language data installed, typically
`tesseract-ocr` and `tesseract-ocr-kor` on Debian/Ubuntu systems. If they are not
available, manifests record `tesseract_ocr_available: false` and the reason in
`tesseract_ocr_error` instead of silently substituting the old template matcher.

## Main Commands

Build data:

```bash
PYTHONPATH=src python -m ryman_font.cli build-dataset \
  --font /usr/share/fonts/truetype/nanum/NanumGothic.ttf \
  --chars "가나다라마바사아자차카타파하한글에코폰트" \
  --out-dir data/hangul \
  --target-savings 0.35,0.45,0.55
```

Train:

```bash
PYTHONPATH=src python -m ryman_font.cli train --config configs/hangul.yaml
```

Infer:

```bash
PYTHONPATH=src python -m ryman_font.cli infer \
  --checkpoint runs/hangul/checkpoint_best.pt \
  --font /usr/share/fonts/truetype/nanum/NanumGothic.ttf \
  --chars "가나다라마바사아자차카타파하한글에코폰트" \
  --out-dir outputs/hangul_045 \
  --target-saving 0.45 \
  --export-ttf outputs/ryman_hangul_045.ttf
```

## Notes

The model is trained from generated pseudo labels. These labels are intentionally
Ryman-like but are not copies of Ryman Eco. They are produced algorithmically
from local glyph geometry.

The current TTF exporter is an MVP contour replacement path. Use PNG previews and
metrics as the primary research outputs.
