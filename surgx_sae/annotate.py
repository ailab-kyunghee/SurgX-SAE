"""Annotation of a unit (a neuron or a latent) with concepts.

A unit is described by its representative sequences on the training videos, and it receives the concepts
whose text embeddings are closest to the embeddings of these sequences.
"""
import numpy as np


def representative_runs(x_per_video, tau):
    """Representative frames of one unit, merged into runs.

    x_per_video : list of 1-D arrays, the value of the unit at every frame of each training video.
    In every video with a positive maximum, the frames with x >= x_max - tau (x_max - x_min) are
    representative, and consecutive representative frames are merged.
    Returns a list of (video_index, first_frame, last_frame).
    """
    runs = []
    for vi, x in enumerate(x_per_video):
        mx, mn = float(x.max()), float(x.min())
        if mx <= 0: continue
        idx = np.where(x >= mx - tau * (mx - mn))[0]
        s = e = int(idx[0])
        for t in idx[1:]:
            t = int(t)
            if t == e + 1: e = t
            else: runs.append((vi, s, e)); s = e = t
        runs.append((vi, s, e))
    return runs


def sample_sequence(first, last, n_frames=16):
    """Frame indices of the representative sequence of a run.

    The sequence spans from 15 frames before the run to its last frame, sampled evenly to n_frames frames.
    """
    lo = max(0, first - (n_frames - 1))
    return [int(round(t)) for t in np.linspace(lo, last, n_frames)]


def rank_concepts(sequence_embeddings, concept_embeddings):
    """Score the concepts for one unit.

    sequence_embeddings : (R, e) unit-norm visual embeddings of the unit's representative sequences.
    concept_embeddings  : (C, e) unit-norm text embeddings of the concept set.
    The score of a concept is its mean cosine similarity to the sequences.
    Returns (order, gap): concept indices by decreasing score, and for each of them
    (s_max - s) / (s_max - s_min), so that the concepts at any theta are a prefix.
    """
    s = (np.asarray(sequence_embeddings) @ np.asarray(concept_embeddings).T).mean(0)
    order = np.argsort(-s)
    gap = (s.max() - s[order]) / max(s.max() - s.min(), 1e-12)
    return order, gap


def concepts_at(ranking, names, theta):
    """Concepts with s >= s_max - theta (s_max - s_min). `ranking` is the output of rank_concepts, or None."""
    if ranking is None: return []
    order, gap = ranking
    return [names[int(i)] for i in order[gap <= theta + 1e-12]]
