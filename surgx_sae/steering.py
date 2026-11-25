"""Steering by concept: turning on the units annotated with a concept that represents a phase."""
import numpy as np


def typical_magnitude(values):
    """Mean value of every unit over the frames on which it is positive (0 for a unit that never is)."""
    V = np.asarray(values); pos = V > 1e-6; cnt = pos.sum(0)
    return np.where(cnt > 0, (V * pos).sum(0) / np.maximum(cnt, 1), 0.0)


def steer_by_concept(logits, predicted, target_phase, units, typical, directions, W, scale, reference):
    """Fraction of frames whose prediction moves to the target phase.

    logits      : (T, P) phase logits of the test frames.
    predicted   : (T,) predictions. Only frames not predicted as the target phase are counted.
    units       : indices of the units annotated with the concept.
    typical     : (U,) typical magnitude of every unit on the training frames.
    directions  : (U, n) direction of every unit in the explained activation, the identity for neurons and
                  the decoder directions for latents.
    W           : (P, n) weight of the final linear layer.
    scale       : size of the edit in units of `reference` (s in the paper).
    reference   : mean distance of a test activation from the median training activation.
    The edit adds every unit at its typical magnitude and is rescaled to the norm scale * reference.
    """
    delta = np.asarray(typical)[units] @ np.asarray(directions)[units]
    delta = scale * reference * delta / np.linalg.norm(delta)
    keep = np.asarray(predicted) != target_phase
    return float(((np.asarray(logits)[keep] + np.asarray(W) @ delta).argmax(1) == target_phase).mean())
