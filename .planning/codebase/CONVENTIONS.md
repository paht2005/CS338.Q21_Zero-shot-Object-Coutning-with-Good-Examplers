# Conventions — Zero-shot Object Counting with Good Exemplars

## Code Style

- **No linter/formatter configured** — no `flake8`, `black`, `ruff`, `isort` config files
- **No type hints** — functions have no type annotations (except a few in `prompt_enhancer.py` docstrings)
- Makefile has a minimal `lint` target that only runs `py_compile` on 4 files
- **Inconsistent formatting** — mix of:
  - Single and double quotes
  - Trailing commas sometimes present, sometimes not
  - Variable indentation in argument lists

## Naming

- **Files:** `snake_case.py` — mostly consistent, some with uppercase (`FSC_train.py`, `FSC_test.py`)
- **Classes:** `PascalCase` — `SupervisedMAE`, `ClipClassifier`, `CrossAttentionBlock`, `TestData`
- **Functions:** `snake_case` — `load_counting_model()`, `enhance_prompt_with_gemini()`, `is_valid_patch()`
- **Constants:** `UPPER_CASE` — `MAX_HW`, `IM_NORM_MEAN`, `BOX_THRESHOLD`, `DENSITY_SCALE`
- **Variables:** `snake_case` — but some inconsistency with single-letter vars (`p`, `f`, `x`)

## Patterns

### Global State
Many scripts use **module-level globals** for models and configuration:
```python
# grounding_pos.py — globals at module level
device = "cuda" if torch.cuda.is_available() else "cpu"
BOX_THRESHOLD = 0.05
clip_model_b32, preprocess_b32 = clip.load("ViT-B/32", device)
binary_classifier = ClipClassifier(clip_model_b32).to(device)
```

### argparse for CLI
Training/test scripts use extensive `argparse` with sensible defaults:
```python
parser.add_argument('--batch_size', default=26, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--lr', type=float, default=None)
```

### Model Loading Pattern
Consistent `torch.load()` + `load_state_dict()` pattern:
```python
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()
```

### Streamlit Caching
Demo uses `@st.cache_resource` for expensive model loading:
```python
@st.cache_resource
def load_all_models():
    ...
```

### Monkey-patching pathlib
Windows compatibility hack appears in multiple files:
```python
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
```

## Error Handling

- **Minimal** — most scripts have no try/except blocks
- Exemplar extraction scripts silently fall back to dummy boxes `[[0,0,20,20]] * 3` when no detections
- `prompt_enhancer.py` has retry logic (up to `max_retries=2`) for Gemini API calls
- Demo app catches prompt enhancement failures and shows `st.warning()`
- `grounding_pos.py` prints `WARNING:` messages when weights or prompt files are missing but continues

## Imports

- No `__init__.py` in root or `code/source-code/` — not importable as packages
- Utility imports use relative paths: `from util.FSC147 import transform_train, transform_val`
- Demo imports assume same-directory: `from demo_inference import load_counting_model`
- GroundingDINO imports use its package structure: `from GroundingDINO.groundingdino.util.inference import ...`

## Documentation

- Good **README.md** files at multiple levels (root, code/, demo/, configs/, scripts/, experiments/, docs/)
- Comprehensive root README with project overview, method details, results, and usage
- `docs/RESULTS.md` — numerical results with provenance to wandb runs
- LaTeX report at `docs/report/main.tex` and slides at `docs/slide/main.tex`
- Inline comments are sparse — mostly in model definition files
- Docstrings only in `prompt_enhancer.py` and `demo_visualization.py`

## Git

- Comprehensive `.gitignore` covering:
  - Large model weights (`*.pth`, `*.pt`, `*.ckpt`, `*.onnx`)
  - Dataset directories, output directories
  - Python cache, virtual environments
  - W&B logs, Jupyter checkpoints
  - OS files (`.DS_Store`, `Thumbs.db`)
- Commits are direct to main (no branch-based workflow visible)
- Large artifacts hosted on OneDrive, referenced from README
