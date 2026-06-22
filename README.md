# EcoFont AI Lab

Cherokee/Korean font files can be rendered into glyph images, converted into
ink-saving pseudo-labels with a rule optimizer, and used to train a lightweight
U-Net that predicts eco masks.

This implementation follows the highest-probability MVP path from
`eco_font_guide.pdf`:

1. Render glyphs from TTF/OTF files.
2. Generate rule-based dot/stripe/center-cut candidates.
3. Pick the best candidate with SSIM, ink ratio, target-saving, and topology
   penalties.
4. Train a Glyph-to-EcoMask U-Net on those pseudo-labels.
5. Run inference to create mask/preview images and JSON metrics.

## Quick Start

Use the current Python environment:

```bash
python -m ecofont.cli inspect-font --font /path/to/font.ttf --language ko
python -m ecofont.cli build-dataset \
  --font /path/to/font.ttf \
  --language ko \
  --output data/ko-demo \
  --targets 0.15,0.25,0.35 \
  --image-size 128
python -m ecofont.cli train \
  --dataset data/ko-demo \
  --output runs/ko-demo/model.pt \
  --epochs 10 \
  --batch-size 16
python -m ecofont.cli infer \
  --font /path/to/font.ttf \
  --checkpoint runs/ko-demo/model.pt \
  --language ko \
  --text "한글 에코폰트 테스트" \
  --output outputs/ko-demo \
  --target-saving 0.25
```

For Cherokee, use a font that actually contains Cherokee glyphs:

```bash
python -m ecofont.cli inspect-font --font /path/to/cherokee-font.ttf --language chr
python -m ecofont.cli build-dataset --font /path/to/cherokee-font.ttf --language chr --output data/chr-demo
```

If you only want the optimized rule baseline without a trained checkpoint:

```bash
python -m ecofont.cli infer \
  --font /path/to/font.ttf \
  --method rules \
  --language ko \
  --text "가나다라마" \
  --output outputs/rule-demo \
  --target-saving 0.25
```

## OCR-Guided Rule Search

When SSIM/topology preservation makes the result too conservative, train a
local OCR-surrogate recognizer and use `ocr-rules`. This loss uses recognizer
confidence plus target ink saving, and can reward outline-changing candidates:

```bash
python -m ecofont.cli train-ocr \
  --font /path/to/cherokee-font.ttf \
  --language chr \
  --output runs/chr-ocr/ocr.pt

python -m ecofont.cli infer \
  --font /path/to/cherokee-font.ttf \
  --method ocr-rules \
  --ocr-checkpoint runs/chr-ocr/ocr.pt \
  --language chr \
  --text "ᎣᏏᏲ ᏣᎳᎩ" \
  --output outputs/chr-ocr-guided \
  --target-saving 0.25 \
  --outline-reward-weight 0.60
```

## Optional Isolated Environment

The repository includes a setup script that keeps the virtual environment and
pip cache under this folder:

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

## Outputs

Dataset builds create:

- `metadata.jsonl`: one row per glyph/target pseudo-label
- `summary.json`: coverage and average metric summary
- `samples/*.npz`: model features, pseudo-label mask, original glyph, eco glyph

Inference creates:

- `metrics.json`: per-glyph and average tradeoff metrics
- `preview.png`: contact sheet comparing original, eco, and removal mask
- `glyphs/*.png`: per-glyph original/eco/mask images

## Notes

- OCR is intentionally an optional evaluator. The local environment does not
  currently include Tesseract, so the core pipeline uses SSIM, topology, and ink
  area. OCR can be added behind the same metrics interface later.
- This MVP predicts raster eco masks. True TTF outline regeneration is a later
  integration layer because robust boolean operations on font outlines are a
  separate engineering problem.
