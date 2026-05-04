# Testing — Zero-shot Object Counting with Good Exemplars

## Test Framework

**No automated test framework** — no pytest, unittest, or any test runner is configured.

## What Exists

### Manual Evaluation Script
- `code/source-code/FSC_test.py` — evaluates model on FSC-147 test split
  - Computes **MAE** (Mean Absolute Error) and **RMSE** (Root Mean Square Error)
  - Runs full forward pass on test images with exemplar annotations
  - Outputs per-image predictions and aggregate metrics
  - Called via `make eval` or `bash scripts/run_evaluation.sh`

### Evaluation Suite Script
- `scripts/run_evaluation.sh` — runs 3 evaluation variants sequentially:
  1. Baseline (GroundingDINO exemplars)
  2. GroundingDINO + Rich Prompt
  3. YOLO-World
  - Reports MAE/RMSE for each variant

### Minimal Compile Check
- `make lint` — runs `python -m py_compile` on 4 files:
  - `models_crossvit.py`, `models_mae_cross.py`, `FSC_train.py`, `FSC_test.py`
  - Only checks syntax, not logic or style

### Jupyter Notebooks
- `code/source-code/notebooks/test_model_va.ipynb` — interactive model testing
- `experiments/exp2/notebook/test_model_va.ipynb` — experiment-specific testing

## What Does NOT Exist

- No unit tests for individual functions/modules
- No integration tests for the pipeline
- No CI/CD pipeline (no GitHub Actions, no pre-commit hooks)
- No test fixtures or mock data
- No regression tests for model accuracy
- No performance benchmarks beyond the manual evaluation script
- No test configuration (`pytest.ini`, `conftest.py`, `tox.ini`)
- No code coverage tracking

## Testing in Practice

The project relies on:
1. **Manual evaluation** — running `FSC_test.py` and comparing MAE/RMSE to documented baselines
2. **Visual inspection** — `demo_app_advanced.py` provides interactive visual verification
3. **W&B tracking** — experiment metrics logged to Weights & Biases for comparison across runs
4. **Visualization outputs** — density maps and bounding boxes saved to `visualizations/` and `visualizations_test/`

## Mocking

No mocking infrastructure. External dependencies (Gemini API, CLIP, YOLO-World) are always called directly.
