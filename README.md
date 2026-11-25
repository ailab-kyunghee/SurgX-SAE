# SurgX-SAE

**SurgX-SAE: Interpreting Surgical Phase Recognition through Sparse Latent Features**

SurgX-SAE is the journal extension of SurgX (MICCAI 2025).
It explains the predictions of a surgical phase recognizer with surgical concepts attached to the latents of a sparse autoencoder (SAE) instead of the neurons of the recognizer.

## Contents

| Folder | Content |
|---|---|
| [`surgx_sae/`](surgx_sae) | Reference implementation of the method and of the evaluation. |
| [`sae_weights/`](sae_weights) | Trained SAEs of the four recognizer × dataset settings, three seeds each. |
| [`results/`](results) | The numbers behind the tables and figures, and the concepts attached to every unit. |
| [`concept_sets/`](concept_sets) | The two concept sets, the concept–phase assignment used to select the qualitative examples and the ranked concepts used for steering. |
| [`prompts/`](prompts) | The phase words and phase sentences used by Prediction Interpretability. |
| [`videos/`](videos) | Ten explanation videos on held-out test videos. |

## Code

`surgx_sae` requires `numpy`, and `torch` only for training an SAE.

| Module | Content |
|---|---|
| `sae.py` | Top-k SAE: inference with released weights (`TopKSAE`) and training (`train_topk_sae`). |
| `annotate.py` | Representative sequences of a unit (threshold τ) and the concepts attached to it (threshold θ). |
| `explain.py` | Contribution of every unit to the predicted phase and selection of the units of an explanation (threshold α). |
| `evaluate.py` | Prediction Interpretability and the number of unique concepts. |
| `select.py` | Selection of (τ, θ, α) on the validation videos and comparison at a matched number of concepts. |
| `monosemanticity.py` | Monosemanticity score. |
| `steering.py` | Steering by concept. |

The code operates on quantities that are extracted once from the frozen recognizer and from the vision-language model.

- The penultimate activation of the recognizer at every frame, and the weight `W` of its final linear layer.
- The SurgCLIP visual embeddings of 16-frame sequences and the SurgCLIP text embeddings of the concepts, the phase words and the phase sentences, all normalised to unit length.

The same functions are used for neurons and for latents.
For neurons the unit values are the activations and the readout is `W`.
For latents the unit values are the SAE latents and the readout is `W @ W_dec`.

```python
import json
import numpy as np
import surgx_sae as X

sae = X.TopKSAE.load("sae_weights/CausalASFormer_Cholec80_d2048_k4_seed0.npz")
concepts = json.load(open("results/unit_concepts/CausalASFormer_Cholec80.json"))["SurgX-SAE (Latent), seed 0"]

# activations: (T, 64) penultimate activations of one video, W: (7, 64), predicted: (T,)
latents = sae.encode(activations)
explanations = X.explain_video(latents, W @ sae.W_dec, predicted, concepts, alpha=1.0)

# names, Ec: the concept set and its text embeddings, Ew and Es: embeddings of the phase words and sentences
pi = X.prediction_interpretability(explanations, names, Ec, Ew, Es)
```

Annotating the units of a new SAE takes three steps.

```python
runs = [X.representative_runs([f[:, u] for f in train_latents], tau=0.05) for u in range(sae.W_enc.shape[0])]
# embed the frames X.sample_sequence(first, last) of every run with the visual encoder of SurgCLIP -> seq_emb[u]: (R, e)
concepts = [X.concepts_at(X.rank_concepts(seq_emb[u], Ec) if runs[u] else None, names, theta=0.01)
            for u in range(len(runs))]
```

With the released weights this implementation reproduces the values in `results/` (PI to within 0.001).
