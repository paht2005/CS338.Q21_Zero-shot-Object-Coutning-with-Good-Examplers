# `configs/`

Machine-readable run configurations (YAML) consumed by the training and
evaluation scripts under `code/source-code/`.

## Available Configs

| File | Description |
|------|-------------|
| `train_baseline.yaml` | VA-Count baseline training (GroundingDINO exemplars) |
| `train_finetune_dino_prompt.yaml` | Fine-tune with GroundingDINO + Rich Prompt |
| `train_finetune_yolo.yaml` | Fine-tune with YOLO-World exemplars |
| `test_baseline.yaml` | Evaluation on FSC-147 test set |

## Usage

Configs can be referenced via CLI:

```bash
cd code/source-code
python FSC_train.py --config ../../configs/train_baseline.yaml
```

The canonical hyper-parameters are also documented in the report
(`docs/report/Report.pdf` §3.3.3).
