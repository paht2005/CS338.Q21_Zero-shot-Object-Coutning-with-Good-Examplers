### Experiment exp4

This folder stores an additional set of VA-Count / Rich Prompt / YOLO-World
experiments used during development of the CS331 project.

Contents:

- `data/` – local copy or subset of FSC147
- `GroundingDINO/` – GroundingDINO code and weights
- `visualizations/` – qualitative visualizations (predictions, density maps)
- `wandb/` – Weights & Biases logs

The exact configuration for each run (baseline, Rich Prompt, YOLO-World, etc.)
can be recovered from the corresponding `wandb` runs or notebook/code used to
launch the experiment.

This directory is not required to run the main code; it is kept for
reproducibility and for inspecting training histories and visual results.

