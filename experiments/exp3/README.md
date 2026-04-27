### Experiment exp3 (YOLO-World)

This folder stores experiments that involve **YOLO-World** for exemplar extraction.

Contents:

- `data/` – local copy or subset of FSC147 used in these runs
- `GroundingDINO/` – GroundingDINO code and weights (for comparison or hybrid setups)
- `visualizations/`, `visualizations_test/` – qualitative visualizations of predictions
- `wandb/` – Weights & Biases logs (training and evaluation)
- `yolov8x-worldv2.pt` – YOLO-World checkpoint used in this experiment

These runs correspond to the **YOLO-World** and **YOLO-World + Rich Prompt**
configurations discussed in Chapter 2 and the experimental tables of
`docs/report/Report.pdf`.

To reproduce or extend these experiments, use the main codebase under
`code/source-code` and point it to this experiment directory as needed.

