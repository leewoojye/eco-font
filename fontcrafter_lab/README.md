# FontCrafter Lab

This folder is a contained FontCrafter reproduction scaffold for experiments
under `eco-font/fontcrafter_lab`.

## What is exact and what is not

The public paper describes FontCrafter as element-driven artistic font creation
with visual in-context inpainting:

- place an element/reference image next to a blank glyph canvas;
- use FLUX.1-Fill to inpaint the glyph region from the element context;
- add Context-aware Mask Adapter (CMA) shape control;
- use attention redirection for region control and dehallucination;
- optionally run edge repainting for boundary refinement.

As of the checked public materials, I did not find an official FontCrafter
repository, ElementFont download, CMA checkpoint, edge repaint checkpoint, or
attention hook implementation. Because of that, exact paper reproduction is not
possible from public artifacts alone.

This lab therefore provides two runnable paths:

1. `flux-sample`: paper-shaped FLUX.1-Fill in-context pipeline. It constructs the
   element + blank canvas and glyph inpaint masks exactly as the paper describes.
   Use this when Hugging Face access to `black-forest-labs/FLUX.1-Fill-dev` and
   a current `diffusers` build are available.
2. `proxy-sample`: local CPU/GPU-free proxy that mimics the disclosed in-context
   and edge repaint behavior with deterministic image processing. This is not
   the official FontCrafter model, but it lets us create Cherokee samples now.

## Local Cherokee proxy sample

```bash
cd /home/woojye2020/decs_jupyter_lab/eco-font/fontcrafter_lab
bash scripts/run_cherokee_proxy.sh
```

Outputs:

- `outputs/cherokee_blue_stone_proxy/contact_sheet.png`
- `outputs/cherokee_blue_stone_proxy/glyphs/*_fontcrafter_proxy.png`
- `outputs/cherokee_blue_stone_proxy/context_inputs/*_context.png`
- `outputs/cherokee_blue_stone_proxy/context_inputs/*_inpaint_mask.png`
- `outputs/cherokee_blue_stone_proxy/summary.json`

## FLUX.1-Fill path

First accept the model terms on Hugging Face for
`black-forest-labs/FLUX.1-Fill-dev`. The model card says access requires
agreeing to share contact information and accepting the license terms.

Then use a current diffusers environment:

```bash
cd /home/woojye2020/decs_jupyter_lab/eco-font/fontcrafter_lab
python -m pip install -U -r requirements.txt
bash scripts/run_cherokee_flux.sh
```

The script keeps Hugging Face cache inside this folder at `.hf_cache`.
In the current checked environment, `diffusers==0.27.2` is installed and does
not expose `FluxFillPipeline`, so the upgrade step is required before this path
can run.

## Notes on CMA

`src/fontcrafter_lab/cma.py` implements the disclosed CMA block shape:
concatenate mask features to MM-DiT block features, project to hidden dimension
64 with GELU, then project back. It is included so released weights can be
loaded later, but there are no official weights in this folder.

## Sources Checked

- FontCrafter arXiv/CVPR 2026 paper: visual in-context generation with
  FLUX.1-Fill, CMA, attention redirection, and edge repainting.
- FLUX.1-Fill-dev model card: `FluxFillPipeline`, license-gated access, and
  current diffusers usage.
