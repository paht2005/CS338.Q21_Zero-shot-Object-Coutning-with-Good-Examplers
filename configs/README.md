# `configs/`

Reserved for **machine-readable run configurations** (YAML / JSON) consumed by
the training and evaluation scripts under `code/source-code/`. The current
runs all use in-line CLI flags (`--anno_file`, `--output_dir`, `--lr`, ...);
the CS338.Q21 team has not yet migrated those into named YAML files.

When configs are added, follow this convention:

```
configs/
├── train_baseline.yaml          # VA-Count GroundingDINO baseline
├── train_richprompt.yaml        # VA-Count + Rich Prompt
├── train_yolo.yaml              # VA-Count + YOLO-World
├── train_yolo_richprompt.yaml   # VA-Count + YOLO-World + Rich Prompt
└── eval_test.yaml               # FSC-147 test-set MAE/RMSE evaluation
```

For now, the canonical hyper-parameters are documented in the report
(`docs/report/Report.pdf` Bảng `tab:hyperparams` and §3.3.3).
