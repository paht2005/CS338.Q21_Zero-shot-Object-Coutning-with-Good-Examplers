# Stack — Zero-shot Object Counting with Good Exemplars

## Language & Runtime

- **Python 3.9+** — primary and only language
- No packaging (no `setup.py`, `pyproject.toml`, or `setup.cfg`) — scripts are run directly

## Core Frameworks

| Framework | Version | Purpose |
|-----------|---------|---------|
| PyTorch | >=2.0, <3.0 | Deep learning backbone (training, inference, model definitions) |
| torchvision | >=0.15 | Image transforms, pretrained backbones |
| timm | >=0.4.9, <1.0 | Vision Transformer utilities (PatchEmbed, Block), version pinned to 0.4.5–0.4.9 in training scripts |
| Streamlit | >=1.30 | Interactive demo web application |
| OpenAI CLIP | git+https://github.com/openai/CLIP.git | Image-text similarity scoring, binary classifier backbone |
| ultralytics | >=8.0 | YOLO-World object detection for exemplar extraction |
| GroundingDINO | vendored (subdir) | Open-set object detection for exemplar extraction |
| transformers | >=4.30 | HuggingFace transformers (used by GroundingDINO internals) |

## Key Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.23, <2.0 | Array operations, density maps |
| scipy | >=1.10 | Image filtering (ndimage), scientific computing |
| pandas | >=1.5 | Data manipulation, CSV/annotation handling |
| matplotlib | >=3.6 | Visualization, density map plotting |
| Pillow | >=9.3 | Image I/O and manipulation |
| opencv-python | >=4.7 | Image processing, bounding box drawing, heatmaps |
| imgaug | >=0.4.0 | Data augmentation during training |
| scikit-learn | >=1.0 | Train/test split, evaluation utilities |
| wandb | latest | Experiment tracking (W&B) |
| tqdm | latest | Progress bars |
| inflect | >=6.0.0 | Noun singularization for prompt generation |
| google-generativeai | >=0.3.0 | Gemini API for Rich Prompt generation |
| python-dotenv | >=1.0.0 | Environment variable loading from `.env` |

## Model Weights (external, not in git)

| Weight File | Source |
|-------------|--------|
| `checkpoint_FSC.pth` | VA-Count pretrained baseline |
| `checkpoint__finetuning_dino_prompt.pth` | Fine-tuned with GroundingDINO + Rich Prompt exemplars |
| `checkpoint__finetuning_yolo.pth` | Fine-tuned with YOLO-World exemplars |
| `yolov8x-worldv2.pt` | YOLO-World v2 weights (duplicated in `code/source-code/`, `demo/`, `experiments/exp3/`) |
| `best_model.pth` | Binary classifier (CLIP-based single-object filter) |
| GroundingDINO weights | Via vendored `GroundingDINO/` directory |
| MAE pretrained weights | `mae_pretrain_vit_base_full.pth` for pretraining stage |

## Configuration

- **YAML configs** in `configs/` — define training/eval hyperparameters per experiment variant
- **Environment variables** via `.env` (template at `env.template`):
  - `GEMINI_API_KEY` — required for Rich Prompt generation
  - `GEMINI_MODEL_NAME` — optional Gemini model override (default: `gemini-2.0-flash-exp`)
  - `WANDB_API_KEY` — optional for experiment tracking
  - `CUDA_VISIBLE_DEVICES` — optional GPU selection
- **argparse** in every training/test script — command-line arguments for paths, hyperparams
- **Makefile** at repo root — convenience targets (`make train-baseline`, `make demo`, `make eval`, etc.)

## Dependency Management

- Two `requirements.txt` files:
  - Root `requirements.txt` — modern PyTorch >=2.0, for general use
  - `code/source-code/requirements-cuda116.txt` — pinned CUDA 11.6 versions to reproduce reported numbers
- `demo/requirements.txt` — identical to root, for standalone demo deployment
- No virtual environment tooling configured (manual conda/venv setup via `scripts/setup_env.sh`)
- GroundingDINO installed as editable package: `pip install -e .` inside its vendored directory

## Hardware / Compute

- Training/eval assumes **NVIDIA GPU with CUDA** (RTX 4060 used in experiments)
- Demo supports **CPU, MPS (macOS), and CUDA**
- Input images resized to **384×384** (`MAX_HW = 384`)
