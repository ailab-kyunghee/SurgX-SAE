# SAE weights

Top-k sparse autoencoders trained on the penultimate activations of the frozen recognizers, at the configuration (d, k) selected for each setting, for three seeds.

Files are named `<recognizer>_<dataset>_d<d>_k<k>_seed<seed>.npz` and contain

| Array | Shape | Meaning |
|---|---|---|
| `W_enc` | (d, n) | encoder weight |
| `b_enc` | (d,) | encoder bias |
| `W_dec` | (n, d) | decoder weight |
| `b_pre` | (n,) | bias subtracted before encoding and added after decoding |
| `k` | scalar | number of active latents |

Here n is the number of neurons of the explained layer (64 for Causal ASFormer, 32 for TeCNO).

```python
import numpy as np

w = np.load("CausalASFormer_Cholec80_d2048_k4_seed0.npz")

def encode(a):                      # a: (N, n) activations
    z = (a - w["b_pre"]) @ w["W_enc"].T + w["b_enc"]
    k = int(w["k"])
    idx = np.argpartition(-z, k - 1, axis=1)[:, :k]
    f = np.zeros_like(z)
    np.put_along_axis(f, idx, np.maximum(np.take_along_axis(z, idx, axis=1), 0), axis=1)
    return f                        # (N, d) latents

def decode(f):
    return f @ w["W_dec"].T + w["b_pre"]
```
