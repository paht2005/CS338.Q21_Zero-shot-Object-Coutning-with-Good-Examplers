# Architecture — Zero-shot Object Counting with Good Exemplars

## Overall Pattern

**Research pipeline architecture** — a multi-stage computer vision system with:
1. **Exemplar extraction stage** (offline, batch) — detect and select representative object patches
2. **Counting stage** (online) — use exemplars to produce density maps and counts
3. **Demo stage** (interactive) — Streamlit UI wrapping the full pipeline

No services, no microservices, no APIs. Scripts communicate through **files on disk** (JSON annotations, text prompts, image files, model checkpoints).

## Pipeline Data Flow

```
Input Image + Text Prompt
       │
       ├─── Prompt Enhancement ──→ Gemini API → Rich Prompt (positive + negative)
       │    (prompt_enhancer.py)
       │
       ├─── Exemplar Detection ──→ GroundingDINO or YOLO-World
       │    (grounding_pos/neg.py, yolo_pos/neg.py)
       │    │
       │    ├── Binary Classifier Filter (CLIP ViT-B/32 + FC → single-object filter)
       │    ├── CLIP Re-ranking (ViT-L/14 → top-k by similarity)
       │    └── Deduplication (IoU-based)
       │
       ├── Output: Positive exemplar patches + Negative exemplar patches
       │
       └─── Counting Model ──→ MAE-based encoder + CrossViT decoder
            (models_mae_cross.py)
            │
            ├── Image encoder (ViT-Base, patch 16, 384×384)
            ├── Exemplar encoder (CNN: Conv→Pool→Conv→Pool→Conv→Pool→Conv→AdaptivePool)
            ├── Cross-attention blocks (exemplar tokens attend to image tokens)
            ├── Density map decoder (Conv layers → 1-channel density map)
            │
            └── Output: Positive density map (Dp) + Negative density map (Dn)
                        Count = sum(Dp) - sum(Dn), scaled by DENSITY_SCALE=60
```

## Key Modules

### Model Definitions
- `models_mae_cross.py` — **SupervisedMAE**: the core counting model
  - MAE ViT-Base encoder (PatchEmbed → Transformer blocks → LayerNorm)
  - CNN exemplar encoder (4-stage Conv+InstanceNorm+ReLU+Pool)
  - Cross-attention decoder (CrossAttentionBlock from `models_crossvit.py`)
  - Density map regression head (4 Conv+GroupNorm+ReLU layers → 1 channel)
- `models_crossvit.py` — CrossAttentionBlock, Attention, Mlp, DropPath
- `models_mae_noct.py` — Variant without cross-attention (ablation)

### Exemplar Extraction
- `grounding_pos.py` — GroundingDINO + CLIP re-ranking for **positive** exemplars
- `grounding_neg.py` — GroundingDINO for **negative** exemplars
- `yolo_pos_withPrompt.py` — YOLO-World + Rich Prompt for positive exemplars
- `yolo_pos_withoutPrompt.py` — YOLO-World without Rich Prompt
- `yolo_neg.py` — YOLO-World for negative exemplars
- `biclassify.py` — Train the CLIP-based binary classifier (single-object filter)

### Training & Evaluation
- `FSC_pretrain.py` — Pretrain baseline VA-Count on FSC-147
- `FSC_train.py` — Fine-tune with enhanced exemplar annotations
- `FSC_test.py` — Evaluate MAE/RMSE on test split

### Demo
- `demo_app_advanced.py` — Streamlit UI entry point
- `demo_inference.py` — Model loading + full inference pipeline
- `demo_pipeline_advanced.py` — Wrapper with adjustable thresholds
- `demo_visualization.py` — Bounding box drawing, density heatmaps, overlays
- `prompt_enhancer.py` — Gemini-based Rich Prompt generation

### Utilities (`util/`)
- `FSC147.py` — Dataset class, transforms (train/val), data augmentation (imgaug)
- `misc.py` — Distributed training helpers, gradient scaler, logging
- `pos_embed.py` — Sinusoidal positional embeddings
- `lr_sched.py`, `lr_decay.py` — Learning rate scheduling
- `crop.py`, `datasets.py`, `lars.py` — Additional utilities

## Entry Points

| Entry Point | How to Run | Purpose |
|-------------|------------|---------|
| `FSC_pretrain.py` | `python FSC_pretrain.py [args]` | Pretrain baseline |
| `FSC_train.py` | `python FSC_train.py [args]` | Fine-tune with exemplars |
| `FSC_test.py` | `python FSC_test.py [args]` | Evaluate model |
| `demo_app_advanced.py` | `streamlit run demo_app_advanced.py` | Launch demo UI |
| `grounding_pos.py` | `python grounding_pos.py` | Generate GroundingDINO positive exemplars |
| `yolo_pos_withPrompt.py` | `python yolo_pos_withPrompt.py` | Generate YOLO-World positive exemplars |
| `biclassify.py` | `python biclassify.py` | Train binary classifier |

## Abstractions & Patterns

- **No class hierarchy** — models are simple `nn.Module` subclasses, no inheritance trees
- **Global variables** — device, thresholds, and model instances are module-level globals in exemplar scripts
- **argparse** — all training/test scripts use argparse with extensive defaults
- **Streamlit caching** — `@st.cache_resource` for model loading in demo
- **Monkey-patching** — `pathlib.PosixPath = pathlib.WindowsPath` used in multiple files for Windows compat
