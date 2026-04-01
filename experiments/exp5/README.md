### Experiment exp5

This folder stores another set of experiments related to the CS331 project
“Zero-shot Object Counting with Good Exemplars”.

Contents:

- `data/` – local copy or subset of FSC147
- `GroundingDINO/` – GroundingDINO code and weights
- `visualizations/` – qualitative visualizations of predictions
- `wandb/` – Weights & Biases logs

Like `exp4`, this directory is used to archive runs and artifacts rather than to
provide the primary runnable code. To run or extend the project, use the main
implementation under `code/cs331-source-code` and, if needed, point it to this
folder for loading existing checkpoints or data.

