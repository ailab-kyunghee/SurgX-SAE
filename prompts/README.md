# Phase prompts

`phase_prompts.json` contains the text with which each surgical phase is represented in Prediction Interpretability (PI).

For each dataset the file gives

- `phases`, the phase names in label order,
- `phase_words`, the phase word, and
- `phase_sentences`, one sentence that describes the phase.

Both are embedded by the text encoder of SurgCLIP.
PI compares the embedding of every concept of an explanation with the embedding of the predicted phase, once against the phase word and once against the phase sentence, and averages the two.

The Cholec80 words and sentences are those of SurgX (MICCAI 2025).
The AutoLaparo sentences follow the same pattern.
