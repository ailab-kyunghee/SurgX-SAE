"""Prediction Interpretability (PI) and the number of unique concepts of a set of explanations."""
import numpy as np


def unique_concepts(explanations):
    """Number of distinct concepts that appear in at least one explanation."""
    return len({c for frame in explanations for u in frame["units"] for c in u["concepts"]})


def prediction_interpretability(explanations, names, concept_embeddings, phase_word_embeddings,
                                phase_sentence_embeddings):
    """PI of the explanations of all frames (the output of explain_video, concatenated over videos).

    Every frame distributes a total weight of one over its concepts: the weight of a unit is its share of the
    positive contributions of the selected units that carry a concept, divided equally among the concepts of
    the unit. The weights are accumulated per predicted phase, and PI is the weighted mean cosine similarity
    between the concept and the predicted phase, computed against the phase word and against the phase
    sentence and averaged. Frames without any concept do not enter.
    """
    idx = {n: i for i, n in enumerate(names)}
    n_phase = len(phase_word_embeddings)
    W = np.zeros((n_phase, len(names)))
    for frame in explanations:
        units = [(max(u["contribution"], 0.0), [c for c in u["concepts"] if c in idx]) for u in frame["units"]]
        units = [(w, cs) for w, cs in units if cs]
        if not units: continue
        total = sum(w for w, _ in units)
        if total <= 0: units = [(1.0, units[0][1])]; total = 1.0
        for w, cs in units:
            for c in cs: W[frame["predicted"], idx[c]] += w / total / len(cs)
    Ec = np.asarray(concept_embeddings)
    sim_word = Ec @ np.asarray(phase_word_embeddings).T
    sim_sent = Ec @ np.asarray(phase_sentence_embeddings).T
    total = W.sum()
    return float(((W * sim_word.T).sum() / total + (W * sim_sent.T).sum() / total) / 2)
