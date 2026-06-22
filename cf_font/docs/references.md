# References

## CF-Font

- Chi Wang, Min Zhou, Tiezheng Ge, Yuning Jiang, Hujun Bao, Weiwei Xu.
  **CF-Font: Content Fusion for Few-Shot Font Generation**. CVPR 2023.
  https://arxiv.org/abs/2303.14017
- Official implementation:
  https://github.com/wangchi95/CF-Font

Implementation mapping in this folder:

- `cf_font.model.ContentFusionModule`: Eq. 2 and Eq. 3 style weight calculation
  and weighted basis-content fusion.
- `cf_font.train.select_basis_fonts`: Eq. 1 style K-medoids basis selection over
  font-level content embeddings.
- `cf_font.losses.projected_character_loss`: PCL with horizontal, vertical, and
  diagonal 1D projections using cumulative distribution distance.
- `cf_font.infer.refine_style_vector`: ISR, optimizing only the font-level style
  vector during inference.

The original CF-Font code depends on DG-Font/DCNv2 and was trained on a large
Chinese font dataset. This folder keeps the CF-Font algorithmic pieces intact
but uses a compact PyTorch generator so the Cherokee eco-font experiment can run
inside this repository.

## Cherokee Font Sources

- Local default: Noto Sans Cherokee v2.001 already present under
  `../assets/fonts/NotoSansCherokee-v2.001`.
- Google Fonts lists Noto Sans Cherokee as a multi-weight family supporting the
  Cherokee script:
  https://fonts.google.com/noto/specimen/Noto+Sans+Cherokee
- Cherokee Nation Language Department lists Cherokee font options including
  Noto Sans Cherokee and Aboriginal Sans:
  https://language.cherokee.org/fonts-and-keyboards/cherokee-fonts/
- `nkCherokee` exists publicly, but its README says it does not use the proper
  Unicode Cherokee page. It is therefore not a safe default for Unicode model
  training:
  https://github.com/neilk/nkCherokee
- Microsoft documents Plantagenet Cherokee as an OpenType Cherokee font supplied
  with Windows/Office. Redistribution is not assumed here:
  https://learn.microsoft.com/en-us/typography/font-list/plantagenet-cherokee
