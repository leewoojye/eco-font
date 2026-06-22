# eco_diff

Ecofont-style glyph optimization and glyph-to-eco-mask training pipeline.

This project is intentionally self-contained under `eco_diff`. It combines
three reference ideas:

- Ecofont experiments: remove small interior regions from glyphs and evaluate
  ink saving versus legibility.
- few-shot font generation pipelines: render TTF fonts into paired glyph image
  datasets before training.
- LF-Font-style locality: preserve local glyph structure with local feature
  channels such as distance transforms, skeletons, coordinates, and target ink
  saving.

The first practical model is not a full diffusion model. It is a U-Net/ResUNet
mask generator trained from rule-generated pseudo labels:

```text
TTF -> glyph render -> rule-based eco mask -> U-Net training -> predicted mask
    -> preview PNGs -> optional contour-based TTF glyph replacement
```

The diffusion branch is available for the more aggressive research setting where
outline distortion is acceptable:

```text
TTF -> glyph render -> rule eco glyph labels -> conditional DDPM
    -> N eco glyph candidates -> OCR/ink/readability evaluator -> best glyph
    -> preview PNGs -> optional contour-based TTF glyph replacement
```

## Install

From this directory:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If you do not want an editable install, use:

```bash
PYTHONPATH=src python -m eco_diff.cli --help
```

## Build A Dataset

Use one or more TTF/OTF files and a charset file:

```bash
PYTHONPATH=src python -m eco_diff.cli build-dataset \
  --font /path/to/font.ttf \
  --charset-file charsets/sample_ascii.txt \
  --out-dir data/ascii_demo \
  --image-size 96 \
  --font-size 76 \
  --target-savings 0.15,0.25,0.35
```

Output:

- `images/`: original rendered glyphs
- `masks/`: pseudo-label cut masks, where white means pixels to remove
- `eco/`: rule-generated eco previews
- `manifest.jsonl`: training records and metrics
- `summary.json`: aggregate dataset metrics

## Train

```bash
PYTHONPATH=src python -m eco_diff.cli train --config configs/default.yaml
```

Important config keys:

- `data.manifest`: dataset manifest path
- `training.output_dir`: checkpoint directory
- `model.input_channels`: default is 6
- `loss.target_saving_weight`: keeps predicted cut area near the requested ink saving

Checkpoints are saved as:

- `checkpoint_last.pt`
- `checkpoint_best.pt`

## Infer

```bash
PYTHONPATH=src python -m eco_diff.cli infer \
  --checkpoint runs/demo/checkpoint_best.pt \
  --font /path/to/font.ttf \
  --chars "ABCDE" \
  --out-dir outputs/demo \
  --target-saving 0.25
```

For a more aggressive preview that forces the predicted cut area to the requested
ink saving target:

```bash
PYTHONPATH=src python -m eco_diff.cli infer \
  --checkpoint runs/demo/checkpoint_best.pt \
  --font /path/to/font.ttf \
  --chars "ABCDE" \
  --out-dir outputs/demo_aggressive \
  --target-saving 0.35 \
  --force-saving
```

Output:

- `original/*.png`
- `mask/*.png`
- `eco/*.png`
- `inference_manifest.jsonl`

Optional TTF export:

```bash
PYTHONPATH=src python -m eco_diff.cli infer \
  --checkpoint runs/demo/checkpoint_best.pt \
  --font /path/to/font.ttf \
  --charset-file charsets/sample_ascii.txt \
  --out-dir outputs/demo_ttf \
  --target-saving 0.25 \
  --export-ttf outputs/demo_ttf/eco.ttf
```

The TTF export is an MVP vectorization path. It replaces selected glyph outlines
from predicted bitmap contours and preserves the original font tables where
possible. Use the PNG previews and metrics as the primary research output.

## Diffusion Candidate Generator

Train the conditional diffusion branch:

```bash
PYTHONPATH=src python -m eco_diff.cli train-diffusion --config configs/diffusion.yaml
```

Sample multiple candidates and select the best with the evaluator:

```bash
PYTHONPATH=src python -m eco_diff.cli sample-diffusion \
  --checkpoint runs/diffusion_smoke/diffusion_best.pt \
  --font /path/to/font.ttf \
  --chars "ECOFONT123" \
  --out-dir outputs/diffusion_demo \
  --target-saving 0.40 \
  --num-candidates 8 \
  --sample-steps 32 \
  --force-ink-budget
```

Optional OCR selection uses a local Tesseract installation:

```bash
PYTHONPATH=src python -m eco_diff.cli sample-diffusion \
  --checkpoint runs/diffusion_smoke/diffusion_best.pt \
  --font /path/to/font.ttf \
  --chars "ABCDE" \
  --out-dir outputs/diffusion_ocr \
  --target-saving 0.40 \
  --num-candidates 8 \
  --force-ink-budget \
  --ocr-lang eng
```

The evaluator always scores ink saving, skeleton recall, topology changes, SSIM,
and a built-in template-OCR fallback over the requested character set. External
OCR is added when the `tesseract` binary and requested language data are
available. Use `--no-template-ocr` only for ablations.

## Smoke Test

```bash
bash scripts/smoke.sh
```

The smoke script finds a system font, builds a tiny dataset, trains for one
epoch, and runs inference.

Diffusion smoke test:

```bash
bash scripts/smoke_diffusion.sh
```

## Notes

- OCR is optional. `eco_diff.ocr` can call a local `tesseract` binary if it is
  installed.
- For Cherokee experiments, start with `charsets/cherokee.txt` and a font that
  actually contains Cherokee glyphs, such as Noto Sans Cherokee.
- The project never needs files outside this directory except input font paths
  that you explicitly pass on the command line.
