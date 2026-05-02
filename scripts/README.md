# `scripts/`

Shell helper scripts that wrap common workflows for reproducibility.

## Available Scripts

| Script | Description |
|--------|-------------|
| `setup_env.sh` | Create conda env and install all dependencies |
| `download_data.sh` | Instructions for downloading FSC147 dataset & checkpoints |
| `run_evaluation.sh` | Run FSC_test.py for all model variants |
| `generate_exemplars.sh` | Generate exemplars with GroundingDINO and/or YOLO-World |

## Usage

```bash
# First-time setup
bash scripts/setup_env.sh

# Run full evaluation
bash scripts/run_evaluation.sh

# Generate exemplars (options: dino, yolo, all)
bash scripts/generate_exemplars.sh all
```

Or use the Makefile from the repo root:

```bash
make help    # See all available commands
make eval    # Run evaluation
make demo    # Launch Streamlit demo
```
