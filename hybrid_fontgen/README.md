# Hybrid Eco Font Generator

Experimental style-conditioned eco-font generator.

This folder is intentionally independent from the sibling `ecofont` and `ryman`
projects. It implements a practical hybrid of:

- localized-expert raster generation inspired by MX-Font style transfer,
- style priors similar to Ryman/contour-line font design,
- OCR-surrogate scoring for readability-guided candidate selection,
- contour-based MVP TTF export inspired by vector-font generation pipelines.

It is not a reproduction of FontDiffuser, MX-Font, or DeepVecFont. It is a small
research MVP that keeps the same engineering shape: raster generation first,
readability constraints second, vector export last.

## Quick Smoke Experiment

From `/home/woojye2020/decs_jupyter_lab/eco-font`:

```bash
bash hybrid_fontgen/scripts/run_cherokee_smoke.sh
```

The script builds a Cherokee pseudo-labeled dataset, trains a style-conditioned
generator, trains an OCR-surrogate recognizer, runs OCR-guided inference, creates
a preview sheet, and exports a small TTF.

## Manual Commands

```bash
cd /home/woojye2020/decs_jupyter_lab/eco-font/hybrid_fontgen
PYTHONPATH=src python -m hybrid_fontgen.cli build-dataset \
  --font ../assets/fonts/NotoSansCherokee-v2.001/NotoSansCherokee/hinted/ttf/NotoSansCherokee-Regular.ttf \
  --chars "ᎠᎡᎢᎣᎤᎥᎦᎧᎨᎩᎪᎫ" \
  --styles contour,centerline,edge,diagonal \
  --target-savings 0.45,0.60 \
  --out-dir data/chr_demo

PYTHONPATH=src python -m hybrid_fontgen.cli train \
  --dataset data/chr_demo \
  --out runs/chr_demo/model.pt \
  --epochs 5 \
  --device cpu

PYTHONPATH=src python -m hybrid_fontgen.cli train-ocr \
  --font ../assets/fonts/NotoSansCherokee-v2.001/NotoSansCherokee/hinted/ttf/NotoSansCherokee-Regular.ttf \
  --chars "ᎣᏏᏲᏣᎳᎩ" \
  --out runs/chr_demo/ocr.pt \
  --device cpu

PYTHONPATH=src python -m hybrid_fontgen.cli infer \
  --checkpoint runs/chr_demo/model.pt \
  --font ../assets/fonts/NotoSansCherokee-v2.001/NotoSansCherokee/hinted/ttf/NotoSansCherokee-Regular.ttf \
  --chars "ᎣᏏᏲᏣᎳᎩ" \
  --style auto \
  --target-saving 0.60 \
  --ocr-checkpoint runs/chr_demo/ocr.pt \
  --out-dir outputs/chr_demo \
  --export-ttf outputs/chr_demo/hybrid_cherokee.ttf \
  --device cpu
```

## Outputs

- `data/*`: generated pseudo labels and metadata.
- `runs/*`: model and OCR-surrogate checkpoints.
- `outputs/*`: original/eco PNGs, `metrics.json`, `preview.png`, and optional TTF.

## Model Summary

Input channels:

1. original glyph
2. normalized distance transform
3. skeleton map
4. selected style prior
5. edge prior
6. target saving rate
7. x-coordinate
8. y-coordinate

The generator is a small U-Net with localized expert convolution blocks. A style
embedding produces gates over the expert branches, allowing different local
filters for contour, centerline, edge, and diagonal styles.

The OCR surrogate is a small CNN glyph classifier trained with synthetic affine
and eco-style augmentations. During inference, `--style auto` generates several
style candidates. OCR is used as a pass/fail gate by default; candidates above
`--ocr-threshold` are ranked by style novelty, aesthetics, and ink saving rather
than by maximizing OCR confidence. Internal stroke-cutting candidates are
included in auto inference and can be emphasized with `--void-style-weight`.
These void candidates use a punch target: they start from the original glyph
bitmap, remove patterned interior pixels, and only then fall back to limited
thinning when holes alone cannot reach the requested saving.
Use `--save-candidates` during inference to write every candidate PNG under
`candidates/`, plus `candidate_sheet.png` and `candidates.json`.
