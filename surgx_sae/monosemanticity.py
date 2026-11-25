"""Monosemanticity score (MS) of Pach et al., 2025."""
import numpy as np


def monosemanticity_score(values, clip_embeddings):
    """MS of every unit.

    values          : (N, U) value of every unit on N clips.
    clip_embeddings : (N, e) unit-norm visual embeddings of the clips.
    The values of a unit are min-max normalised to [0, 1] over the clips, and its MS is the mean cosine
    similarity over all pairs of different clips, weighted by the product of the normalised values.
    Returns (U,) scores, NaN for a unit that is non-zero on fewer than two clips.
    """
    F = np.asarray(values, np.float64); E = np.asarray(clip_embeddings, np.float64)
    lo, hi = F.min(0), F.max(0)
    A = (F - lo) / np.where(hi > lo, hi - lo, 1)
    G = E.T @ A
    sq = (A ** 2).sum(0)
    num = (G ** 2).sum(0) - sq           # sum over pairs i != j of a_i a_j cos(e_i, e_j)
    den = A.sum(0) ** 2 - sq             # sum over pairs i != j of a_i a_j
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
