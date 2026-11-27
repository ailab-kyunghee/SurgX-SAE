# Results

The files give the numbers behind the tables and figures of the paper.
Settings are named `<recognizer>_<dataset>`.
PI is Prediction Interpretability, and for SurgX-SAE every value is the mean over three SAE seeds unless a seed is given.

| File | Content |
|---|---|
| `selected_settings.json` | The explanation thresholds (τ, θ, α) selected on the validation videos for each method, the SAE configuration (d, k), and PI and the number of unique concepts on the validation and the held-out test videos. |
| `matched_budget.json` | For lower bounds of 20, 40, 60 and 80 unique concepts, the configuration with the best validation PI of each method, with its validation and test PI. |
| `monosemanticity.json` | The monosemanticity score (MS) of the neurons and of the latents of every SAE configuration, on 16-frame validation clips, one ending at every frame. |
| `steering.json` | Steering by concept. For each SAE seed and target phase, the concept turned on (the highest-ranked concept attached to at least one unit of both methods), the number of units, and the fraction of test frames steered to the phase at the matched edit sizes s = 1 and s = 2. The neuron values are listed per SAE seed because the shared concept can differ between the SAEs. |
| `reconstruction.json` | Accuracy, agreement with the recognizer and fraction of variance unexplained when the explained activation is replaced by its SAE reconstruction (seed 0) or by the mean activation. |
| `concept_coverage.json` | Number of distinct concepts, concepts per unit and annotated units as a function of θ at τ = 0.001. |
| `unit_concepts/` | The concepts attached to every neuron and every latent at the selected settings. Each list is indexed by unit, and an empty list is a unit without a concept. |
