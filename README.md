# SurgX-SAE

**SurgX-SAE: Interpreting Surgical Phase Recognition through Sparse Latent Features**

SurgX-SAE is the journal extension of SurgX (MICCAI 2025).
It explains the predictions of a surgical phase recognizer with surgical concepts attached to the latents of a sparse autoencoder instead of the neurons of the recognizer.

## Concept sets and prompts

- [`concept_sets/`](concept_sets) contains the two concept sets (ChoLec-270 for Cholec80 and the 336-concept hysterectomy set for AutoLaparo), the concept–phase assignment used to select the qualitative examples and the ranked concepts used for steering.
- [`prompts/`](prompts) contains the phase words and phase sentences used by Prediction Interpretability.
