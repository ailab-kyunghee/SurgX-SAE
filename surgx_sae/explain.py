"""Explanation of a prediction by the concepts of the units that contribute most to the predicted phase."""
import numpy as np


def unit_contributions(values, readout, predicted):
    """Contribution of every unit to the predicted phase of every frame.

    values    : (T, U) unit values, the activations for neurons or the SAE latents.
    readout   : (P, U) weight from every unit to every phase logit. For neurons this is the weight of the
                final linear layer W, and for latents it is W @ W_dec.
    predicted : (T,) predicted phase of every frame.
    """
    return np.asarray(values) * np.asarray(readout)[np.asarray(predicted)]


def select_units(contribution, alpha):
    """Units of one frame with a positive contribution c >= c_max - alpha (c_max - c_min), largest first.

    The maximum and the minimum are taken over the positive contributions.
    When no contribution is positive, the unit with the largest contribution is returned.
    """
    c = np.asarray(contribution)
    pos = np.where(c > 1e-6)[0]
    if pos.size == 0: return np.array([int(np.argmax(c))])
    mx, mn = float(c[pos].max()), float(c[pos].min())
    idx = pos[c[pos] >= mx - alpha * (mx - mn)]
    return idx[np.argsort(-c[idx])]


def explain_video(values, readout, predicted, unit_concepts, alpha):
    """Explanations of all frames of one video.

    unit_concepts : list of length U, the concepts attached to every unit.
    Returns one dict per frame with the predicted phase and the selected units, each with its contribution
    and its concepts.
    """
    C = unit_contributions(values, readout, predicted)
    return [{"predicted": int(predicted[t]),
             "units": [{"unit": int(i), "contribution": float(C[t, i]), "concepts": list(unit_concepts[i])}
                       for i in select_units(C[t], alpha)]}
            for t in range(len(C))]
