### Experiment exp5

This folder stores another set of experiments run by the CS338.Q21 team on
"Zero-shot Object Counting with Good Exemplars" and is kept as part of the
baseline reference set for the project.

Contents:

- `data/` – local copy or subset of FSC147
- `GroundingDINO/` – GroundingDINO code and weights
- `visualizations/` – qualitative visualizations of predictions
- `wandb/` – Weights & Biases logs

Like `exp4`, this directory is used to archive runs and artifacts rather than to
provide the primary runnable code. To run or extend the project, use the main
implementation under `code/source-code` and, if needed, point it to this
folder for loading existing checkpoints or data.

