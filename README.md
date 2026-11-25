# SurgX-SAE

**Interpreting Surgical Phase Recognition through Sparse Latent Features** — journal extension of SurgX (MICCAI 2025). Work in progress.

Project page with current results: **https://ailab-kyunghee.github.io/SurgX-SAE/**

The page is built from experiment outputs by `site_build.py` in the SurgX experiment repository and deployed from the `gh-pages` branch.

## Concept sets and prompts

- [`concept_sets/`](concept_sets) contains the two concept sets (ChoLec-270 for Cholec80 and the 336-concept hysterectomy set for AutoLaparo), the concept–phase assignment used to select the qualitative examples and the ranked concepts used for steering.
- [`prompts/`](prompts) contains the phase words and phase sentences used by Prediction Interpretability.
