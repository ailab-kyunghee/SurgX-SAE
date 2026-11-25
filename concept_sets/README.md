# Concept sets

Each explained unit (a neuron or an SAE latent) is annotated with concepts from the set of the corresponding surgery.
Concepts are embedded as written, without a prompt template, by the text encoder of SurgCLIP.

| File | Dataset | Concepts | Source |
|---|---|---|---|
| `ChoLec-270.json` | Cholec80 (cholecystectomy) | 270 | The concept set of SurgX (MICCAI 2025), extracted from 11 cholecystectomy lecture videos on the WebSurg e-learning platform and four cholecystectomy articles. |
| `Hysterectomy-336.json` | AutoLaparo (hysterectomy) | 336 | Derived from the captions of hysterectomy videos in SurgLaVi-β. It follows the composition of ChoLec-270, in which about half of the concepts describe actions and the others name instruments, anatomy and findings. |

Both files are JSON arrays of strings, in the order used by the experiments.

## Concept–phase assignment

`concept_phase_assignment.json` assigns two lists of phases to every concept of both sets, on the basis of the phase definitions and the concept set only.

- `related_phases` are the phases to which the concept is closely related.
- `possible_phases` are the phases in which the concept can occur.

Phases are given as indices into the `phases` list of the dataset (0 to 6).
The assignment is used only to select the example frames of the qualitative comparison.
It is not used for annotation, for the choice of any setting or for Prediction Interpretability.

## Concepts for steering

`steering_concepts.json` lists, for each phase, the concepts that represent the phase in ranked order.
The ranking was made on the basis of the phase definitions and the concept set only, without access to the recognizer, the annotations or any result.
The concept turned on for a target phase is the highest-ranked concept attached to at least one unit of both methods.
