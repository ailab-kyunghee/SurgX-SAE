"""Selection of the explanation thresholds on the validation videos."""


def select_configuration(rows, quantile=0.2):
    """Among the fraction `quantile` of configurations with the highest validation PI, the one with the most
    unique concepts (ties by PI).

    rows : list of dicts with the keys "pi" and "concepts", one per configuration (tau, theta, alpha),
           for SurgX-SAE averaged over the SAE seeds.
    """
    rows = sorted(rows, key=lambda r: -r["pi"])
    top = rows[:max(1, int(round(quantile * len(rows))))]
    return max(top, key=lambda r: (r["concepts"], r["pi"]))


def best_at_budget(rows, budget):
    """The configuration with the highest validation PI among those with at least `budget` unique concepts."""
    ok = [r for r in rows if r["concepts"] >= budget]
    return max(ok, key=lambda r: (r["pi"], r["concepts"])) if ok else None
