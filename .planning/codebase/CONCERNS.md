# Concerns — Zero-shot Object Counting with Good Exemplars

## Critical Issues

### 1. Massive Code Duplication
**Severity: High** | **Files affected: ~15 files**

The `demo/` directory is a full copy of source files from `code/source-code/`:
- `demo_app_advanced.py`, `demo_inference.py`, `demo_pipeline_advanced.py`
- `demo_visualization.py`, `prompt_enhancer.py`
- `models_mae_cross.py`, `models_crossvit.py`, `util/pos_embed.py`

These files can drift independently. Any bug fix in `code/source-code/` must be manually replicated to `demo/`.

Similarly, `experiments/exp2/util/` and `experiments/exp3/` contain additional copies of utility files.

**Impact:** Bug fixes get lost, behavior diverges between demo and source.

### 2. Hardcoded Paths and Windows Monkey-patch
**Severity: Medium** | **Files affected: `FSC_train.py`, `FSC_test.py`, `demo_inference.py`, `util/FSC147.py`**

```python
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
```

This global monkey-patch breaks on macOS/Linux when loading checkpoints saved on Windows. It's applied unconditionally, even on non-Windows systems.

Multiple scripts also have hardcoded relative paths (`./data/FSC147/`, `./data/out/classify/best_model.pth`) that assume the working directory is `code/source-code/`.

### 3. No Automated Tests
**Severity: High**

Zero unit tests, zero integration tests, no CI. The only validation is running the full evaluation suite (`FSC_test.py`) which requires the complete FSC-147 dataset and GPU. There's no way to quickly verify that a code change doesn't break anything.

### 4. Security: `weights_only=False` in `torch.load()`
**Severity: Medium** | **Files affected: `grounding_pos.py`, `demo_inference.py`, `FSC_test.py`, `FSC_train.py`**

```python
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
```

`weights_only=False` allows arbitrary code execution from pickle files. This is acceptable for trusted local checkpoints but is a risk if untrusted weights are loaded.

### 5. Global Model Loading at Import Time
**Severity: Medium** | **Files affected: `grounding_pos.py`, `yolo_pos_withPrompt.py`, `yolo_pos_withoutPrompt.py`**

Models (CLIP, binary classifier) are loaded at **module import time** as global variables:
```python
clip_model_b32, preprocess_b32 = clip.load("ViT-B/32", device)
binary_classifier = ClipClassifier(clip_model_b32).to(device)
```

This means importing these modules always triggers heavy model loading, even if you only need a utility function. It also prevents proper device management.

## Technical Debt

### 6. Duplicate ClipClassifier Definition
**Severity: Medium** | **Files: `grounding_pos.py`, `yolo_pos_withPrompt.py`, `yolo_pos_withoutPrompt.py`, `demo_inference.py`, `biclassify.py`**

The `ClipClassifier` class is copy-pasted across 5 files with slight variations. There's no shared module for it.

### 7. Dead Code and Commented-Out Lines
**Severity: Low** | **Multiple files**

- `FSC_test.py`: `self.TransformTrain = transform_train(args, do_aug=do_aug)` commented out
- `util/FSC147.py`: `args = get_args_parser()` called at module level (side effect on import)
- Various `print()` debug statements left in production code

### 8. Inconsistent Device Handling
**Severity: Medium**

- Some files hardcode `device = "cuda"` and fail on CPU-only machines
- Demo properly handles `"cuda" if torch.cuda.is_available() else "cpu"`
- `grounding_pos.py` uses `device = "cuda" if torch.cuda.is_available() else "cpu"` at module level but some functions take a `device` parameter defaulting to `'gpu'` (not a valid PyTorch device string)

### 9. YAML Configs Not Used by Scripts
**Severity: Low** | **Files: `configs/*.yaml`**

YAML config files exist in `configs/` but training scripts use `argparse` only. The YAML files document recommended arguments as comments but aren't actually parsed — there's no config-loading code in `FSC_train.py` or `FSC_test.py`.

### 10. Version Pin Issues
**Severity: Low**

- `timm` is pinned to `0.4.5–0.4.9` via runtime assertion in `FSC_train.py` (`assert "0.4.5" <= timm.__version__ <= "0.4.9"`), but `requirements.txt` allows `>=0.4.9, <1.0`
- This mismatch could cause runtime crashes when the requirement is satisfied but the assertion fails

## Performance Concerns

### 11. No Batch Processing in Exemplar Extraction
Exemplar generation scripts (`grounding_pos.py`, `yolo_pos_withPrompt.py`) process images **one at a time** in a loop. For the full FSC-147 dataset (6,146 images), this takes hours.

### 12. Redundant Model Loading in Demo
The demo loads ALL models upfront (counting model, YOLO, GroundingDINO, CLIP, binary classifier), even if the user only uses one detection mode.

## Fragile Areas

- **Exemplar JSON format** — the annotation JSON structure is an implicit contract between generation scripts and training scripts. No schema validation.
- **Path assumptions** — all scripts assume `cwd = code/source-code/`. Running from repo root breaks most things unless you `cd` first.
- **GroundingDINO vendor** — vendored as a subdirectory with its own `setup.py`. Changes upstream require manual sync.
