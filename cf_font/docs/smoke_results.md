# Cherokee Smoke Experiment Results

Run date: 2026-06-19

## Setup

- Fonts: 9 local Noto Sans Cherokee hinted TTF weights.
- Characters: first 24 uppercase Cherokee syllabary glyphs from
  `charsets/cherokee.txt`.
- Dataset: 1,728 samples = 9 fonts x 24 chars x 4 eco styles x 2 target
  savings.
- Model: compact CF-Font implementation with base stage, K-medoids basis
  selection, CFM stage, PCL, and ISR-enabled inference.
- Training: 2 base epochs + 2 CFM epochs on CUDA.
- OCR surrogate: 24 classes, 8 epochs, best validation accuracy 1.0.

## CF-Font Basis Fonts

K-medoids selected these basis fonts:

- Noto Sans Cherokee ExtraBold
- Noto Sans Cherokee Light
- Noto Sans Cherokee Medium
- Noto Sans Cherokee ExtraLight

The full basis weights are in `runs/cherokee_smoke/basis_summary.json`.

## Training

- Checkpoint: `runs/cherokee_smoke/model.pt`
- Best validation loss: 0.8757241169611613
- Training summary: `runs/cherokee_smoke/training_summary.json`
- OCR summary: `runs/cherokee_smoke/ocr_summary.json`

## Inference Trade-Off

All rows use the same text sample `ᎣᏏᏲᏣᎳᎩᎠᎡᎢᎣ`, auto style search, OCR
candidate gating, and 4 ISR steps.

| Target saving | Output folder | Actual ink saving | OCR match | OCR confidence | Skeleton recall | Saving gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.35 | `outputs/cherokee_smoke_s035` | 0.4427 | 0.6667 | 0.4613 | 0.9459 | 0.0927 |
| 0.45 | `outputs/cherokee_smoke_s045` | 0.5446 | 0.6667 | 0.3762 | 0.8710 | 0.0946 |
| 0.60 | `outputs/cherokee_smoke` | 0.6823 | 0.4444 | 0.2197 | 0.6938 | 0.0823 |

Interpretation: higher saving produces more visually novel/aggressive glyphs,
but the OCR surrogate and skeleton preservation drop. The current smoke model's
best readability/saving balance is the 0.35 to 0.45 target range; 0.60 is useful
as an aggressive upper bound rather than a production default.
