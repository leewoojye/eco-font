# CF-Font Eco Cherokee

This folder is a self-contained CF-Font style experiment for Cherokee-script
eco font generation. It does not modify sibling projects.

The implemented research path follows **CF-Font: Content Fusion for Few-Shot
Font Generation**:

1. Train a base content/style generator.
2. Collect content features from the trained content encoder.
3. Select basis fonts with K-medoids over font-level content embeddings.
4. Replace each target content feature with a weighted sum of the same glyph's
   basis-font content features.
5. Continue training with the Content Fusion Module and Projected Character
   Loss.
6. Optionally refine the target font style vector at inference with ISR.

The eco-font layer follows the existing repository's ink-saving/readability
setup: pseudo targets are generated with contour/centerline/edge/diagonal
stroke-retention priors, then inference ranks candidates by ink saving,
skeleton preservation, topology, and optional OCR-surrogate confidence.

## Quick Smoke Run

From `/home/woojye2020/decs_jupyter_lab/eco-font`:

```bash
bash cf_font/scripts/run_cherokee_smoke.sh
```

The script builds a Cherokee dataset from the local Noto Sans Cherokee weights,
trains the base and CF stages, trains a small OCR surrogate, runs inference, and
writes metrics plus preview images under `cf_font/outputs/cherokee_smoke`.

## Manual Commands

```bash
cd /home/woojye2020/decs_jupyter_lab/eco-font/cf_font

PYTHONPATH=src python -m cf_font.cli build-dataset \
  --fonts-glob "../assets/fonts/NotoSansCherokee-v2.001/NotoSansCherokee/hinted/ttf/NotoSansCherokee-*.ttf" \
  --chars-file charsets/cherokee.txt \
  --out-dir data/cherokee \
  --image-size 64 \
  --limit-chars 32

PYTHONPATH=src python -m cf_font.cli train \
  --dataset data/cherokee \
  --out runs/cherokee/model.pt \
  --base-epochs 2 \
  --cf-epochs 2 \
  --basis-count 4 \
  --device auto

PYTHONPATH=src python -m cf_font.cli train-ocr \
  --dataset data/cherokee \
  --out runs/cherokee/ocr.pt \
  --epochs 3 \
  --device auto

PYTHONPATH=src python -m cf_font.cli infer \
  --checkpoint runs/cherokee/model.pt \
  --font "../assets/fonts/NotoSansCherokee-v2.001/NotoSansCherokee/hinted/ttf/NotoSansCherokee-Regular.ttf" \
  --chars "ᎣᏏᏲᏣᎳᎩ" \
  --out-dir outputs/cherokee \
  --target-saving 0.60 \
  --style auto \
  --ocr-checkpoint runs/cherokee/ocr.pt \
  --isr-steps 8 \
  --save-candidates
```

## Outputs

- `data/*`: rendered glyph tensors, eco pseudo labels, metadata.
- `runs/*`: base/CF checkpoint, basis-font selection, OCR checkpoint.
- `outputs/*`: per-glyph PNGs, candidate sheet, preview, and metrics JSON.

## Notes

- The default target is Cherokee script because the surrounding repository uses
  `chr`/`cherokee` assets and already includes Noto Sans Cherokee. If the
  intended target was Circassian Cyrillic instead, swap the charset and font
  list; the CFM training code is script-agnostic.
- `nkCherokee` is documented as not using the proper Unicode Cherokee page, so
  it is not used as a default training font. See `docs/references.md`.
