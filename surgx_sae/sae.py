"""Top-k sparse autoencoder on the explained activation of a frozen phase recognizer.

    f     = ReLU(TopK_k(W_enc (a - b_pre) + b_enc))
    a_hat = W_dec f + b_pre
"""
import numpy as np


class TopKSAE:
    """Inference with trained weights (numpy only)."""

    def __init__(self, W_enc, b_enc, W_dec, b_pre, k):
        self.W_enc = np.asarray(W_enc, np.float32)   # (d, n)
        self.b_enc = np.asarray(b_enc, np.float32)   # (d,)
        self.W_dec = np.asarray(W_dec, np.float32)   # (n, d)
        self.b_pre = np.asarray(b_pre, np.float32)   # (n,)
        self.k = int(k)

    @classmethod
    def load(cls, path):
        w = np.load(path)
        return cls(w["W_enc"], w["b_enc"], w["W_dec"], w["b_pre"], int(w["k"]))

    @property
    def directions(self):
        """Decoder direction of every latent in activation space, shape (d, n)."""
        return self.W_dec.T

    def encode(self, a):
        """a: (N, n) activations -> (N, d) latents with at most k non-zero entries per row."""
        a = np.atleast_2d(np.asarray(a, np.float32))
        z = (a - self.b_pre) @ self.W_enc.T + self.b_enc
        idx = np.argpartition(-z, self.k - 1, axis=1)[:, :self.k]
        f = np.zeros_like(z)
        np.put_along_axis(f, idx, np.maximum(np.take_along_axis(z, idx, axis=1), 0), axis=1)
        return f

    def decode(self, f):
        return np.asarray(f, np.float32) @ self.W_dec.T + self.b_pre

    def fvu(self, a):
        """Fraction of variance unexplained of the reconstruction."""
        a = np.asarray(a, np.float32)
        return float(((self.decode(self.encode(a)) - a) ** 2).sum() / ((a - a.mean(0)) ** 2).sum())


def train_topk_sae(X, d, k, seed, X_val, lr=2e-4, max_epochs=10000, batch_size=8192, aux_coef=1 / 32,
                   eval_every=50, patience=4, min_epochs=1000, device=None):
    """Train a top-k SAE with the auxiliary loss on dead latents (AuxK).

    X, X_val : (N, n) training and validation activations.
    Decoder columns have unit norm, the encoder is initialised as the transpose of the decoder, and b_pre is
    initialised at the geometric median of the training activations.
    A latent is dead when it was not active during the previous epoch, and the k_aux largest dead latents
    reconstruct the residual of the main reconstruction.
    The validation FVU is evaluated every `eval_every` epochs, training stops after `patience` evaluations
    without a new minimum (not before `min_epochs`), and the weights of the minimum are returned.
    """
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    X = torch.tensor(np.asarray(X, np.float32), device=dev); N, n = X.shape
    Xv = torch.tensor(np.asarray(X_val, np.float32), device=dev)
    k_aux = min(d, 2 ** int(round(np.log2(n / 2))))

    med = X.mean(0)                                             # geometric median (Weiszfeld)
    for _ in range(200):
        w = 1.0 / (X - med).norm(dim=1).clamp_min(1e-8); nxt = (w[:, None] * X).sum(0) / w.sum()
        done = (nxt - med).norm() < 1e-7; med = nxt
        if done: break

    W_dec = torch.randn(n, d, device=dev); W_dec /= W_dec.norm(dim=0, keepdim=True)
    W_dec = torch.nn.Parameter(W_dec); W_enc = torch.nn.Parameter(W_dec.detach().t().clone())
    b_enc = torch.nn.Parameter(torch.zeros(d, device=dev)); b_pre = torch.nn.Parameter(med.clone())

    def topk(z, kk, relu=True):
        v, i = torch.topk(z, kk, dim=1)
        out = torch.zeros_like(z); out.scatter_(1, i, torch.relu(v) if relu else v); return out

    with torch.no_grad():                                       # scale the encoder so that the reconstruction norm matches the data
        xs = X[torch.randperm(N, generator=torch.Generator().manual_seed(seed))[:32768].to(dev)] - med
        ff = topk(xs @ W_enc.t(), k)
        W_enc.data /= ((ff @ W_dec.t()).norm(dim=1).mean() / xs.norm(dim=1).mean()).clamp_min(1e-8)
    var_x = ((X - X.mean(0)) ** 2).sum(1).mean()
    opt = torch.optim.Adam([W_enc, b_enc, W_dec, b_pre], lr=lr, eps=6.25e-10)

    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    dead = torch.zeros(d, dtype=torch.bool, device=dev)
    best, bad = (float("inf"), 0, None), 0
    for ep in range(max_epochs):
        perm = torch.randperm(N, generator=g).to(dev); fired = torch.zeros(d, dtype=torch.bool, device=dev)
        for s in range(0, N, batch_size):
            xb = X[perm[s:s + batch_size]]
            z = (xb - b_pre) @ W_enc.t() + b_enc; f = topk(z, k)
            e = xb - (f @ W_dec.t() + b_pre); loss = (e ** 2).sum(1).mean() / var_x
            nd = int(dead.sum())
            if nd > 0 and aux_coef > 0:
                fa = topk(z.masked_fill(~dead, float("-inf")), min(k_aux, nd)); ed = e.detach()
                loss = loss + aux_coef * (((ed - fa @ W_dec.t()) ** 2).sum(1).mean()
                                          / ((ed - ed.mean(0)) ** 2).sum(1).mean().clamp_min(1e-12))
            opt.zero_grad(); loss.backward()
            with torch.no_grad():                               # keep the decoder columns at unit norm
                W_dec.grad -= (W_dec.grad * W_dec.data).sum(0, keepdim=True) * W_dec.data
            opt.step()
            with torch.no_grad():
                W_dec.data /= (W_dec.data.norm(dim=0, keepdim=True) + 1e-8); fired |= (f > 0).any(0)
        dead = ~fired
        if (ep + 1) % eval_every == 0:
            with torch.no_grad():
                rec = topk((Xv - b_pre) @ W_enc.t() + b_enc, k) @ W_dec.t() + b_pre
                fv = (((rec - Xv) ** 2).sum() / ((Xv - Xv.mean(0)) ** 2).sum()).item()
            if fv < best[0]: best = (fv, ep + 1, [t.detach().clone() for t in (W_enc, b_enc, W_dec, b_pre)]); bad = 0
            else:
                bad += 1
                if bad >= patience and ep + 1 >= min_epochs: break
    We, be, Wd, bp = (t.cpu().numpy() for t in (best[2] or [t.detach() for t in (W_enc, b_enc, W_dec, b_pre)]))
    return TopKSAE(We, be, Wd, bp, k), {"val_fvu": best[0], "epoch": best[1]}
