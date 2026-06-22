# eco_research_hangul

Research-oriented Hangul eco typeface experiment.

This folder intentionally avoids hand-crafted pseudo eco masks. Training targets
come from real Hangul eco fonts installed on the machine, such as
`NanumGothicEco` and `NanumMyeongjoEco`. The generator is a conditional DDPM
inspired by diffusion-based font generation papers. Evaluation follows the
ecofont literature: ink coverage/saving plus OCR readability.

## References Used As Implementation Guides

- Ryman Eco: line-based sustainable typography, balancing ink saving, legibility,
  and visual appeal.
- Ecofont toner-consumption studies: evaluate ink/toner reduction and visual
  readability instead of only pixel similarity.
- Diff-Font/DDPM: conditional diffusion model predicts the target eco glyph from
  a source glyph condition.

## Quick Run

```bash
cd /home/woojye2020/decs_jupyter_lab/eco-font/eco_research_hangul
../.venv/bin/python -m eco_research_hangul.cli build-dataset --config configs/smoke.yaml
../.venv/bin/python -m eco_research_hangul.cli train --config configs/smoke.yaml
../.venv/bin/python -m eco_research_hangul.cli infer --config configs/smoke.yaml
../.venv/bin/python -m eco_research_hangul.cli report --root outputs/smoke --output outputs/smoke/contact_sheet.png
```

The smoke config is intentionally small. It proves the pipeline and produces a
small Hangul sample, not a production-quality font.

## OCR/Ink/Style Guided Run

The `guided-infer` path is the experiment for moving away from the perforated
Nanum Eco target. The trained diffusion model is used as one candidate source,
but the final glyph is selected by objective scores:

- Tesseract Korean OCR exact match gate (`--psm 8` in `configs/guided.yaml`)
- ink coverage reduction, following ecofont evaluation practice
- VGG Gram-style distance to the downloaded Ryman Eco font reference
- design gates that reject small dot holes and foreground fragments
- mild inline engraving candidates that keep strokes continuous

```bash
cd /home/woojye2020/decs_jupyter_lab/eco-font/eco_research_hangul
../.venv/bin/eco-research-hangul guided-infer --config configs/guided.yaml
../.venv/bin/eco-research-hangul report --root outputs/guided --output outputs/guided/contact_sheet.png
```

Current guided smoke result:

- `outputs/guided/contact_sheet.png`
- `outputs/guided/metrics_summary.json`
- mean ink saving: about 36%
- Tesseract exact-match accuracy: 100% on the 20-character smoke set
- selected candidates have zero small-hole and zero small-fragment design-gate
  counts
