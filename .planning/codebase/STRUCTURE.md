# Structure — Zero-shot Object Counting with Good Exemplars

## Directory Layout

```
cs338-zero-shot-object-counting-with-good-examplers/
│
├── README.md                          # Main project documentation
├── Makefile                           # Top-level build/run shortcuts
├── LICENSE                            # MIT License
├── requirements.txt                   # Root Python dependencies
├── env.template                       # .env template (Gemini key, wandb, CUDA)
├── .gitignore                         # Comprehensive gitignore
│
├── code/                              # Main implementation
│   ├── README.md
│   └── source-code/                   # All source code lives here
│       ├── models_mae_cross.py        # Core counting model (SupervisedMAE)
│       ├── models_crossvit.py         # Cross-attention blocks
│       ├── models_mae_noct.py         # MAE variant without cross-attention
│       ├── FSC_pretrain.py            # Pretraining script
│       ├── FSC_train.py               # Training / fine-tuning script
│       ├── FSC_test.py                # Evaluation script
│       ├── prompt_enhancer.py         # Gemini Rich Prompt generation
│       ├── biclassify.py              # Binary classifier training
│       ├── grounding_pos.py           # GroundingDINO positive exemplars
│       ├── grounding_neg.py           # GroundingDINO negative exemplars
│       ├── yolo_pos_withPrompt.py     # YOLO-World + prompt positives
│       ├── yolo_pos_withoutPrompt.py  # YOLO-World no-prompt positives
│       ├── yolo_neg.py                # YOLO-World negatives
│       ├── demo_app_advanced.py       # Streamlit demo UI
│       ├── demo_inference.py          # Model loading + inference pipeline
│       ├── demo_pipeline_advanced.py  # Pipeline wrapper with thresholds
│       ├── demo_visualization.py      # Visualization utilities
│       ├── inference_official.py      # Official inference script
│       ├── datasetmake.py             # Dataset preparation utility
│       ├── create_txt.ipynb           # Notebook for text file generation
│       ├── requirements.txt           # Python dependencies (duplicate of root)
│       ├── requirements-cuda116.txt   # Pinned CUDA 11.6 dependencies
│       ├── yolov8x-worldv2.pt         # YOLO-World weights (gitignored)
│       │
│       ├── util/                      # Shared utilities
│       │   ├── __init__.py
│       │   ├── FSC147.py              # Dataset class + transforms
│       │   ├── misc.py                # Distributed training, gradient scaling
│       │   ├── pos_embed.py           # Positional embeddings
│       │   ├── lr_sched.py            # LR scheduling
│       │   ├── lr_decay.py            # LR decay
│       │   ├── crop.py                # Cropping utilities
│       │   ├── datasets.py            # Additional dataset utils
│       │   └── lars.py                # LARS optimizer
│       │
│       ├── data/                      # Dataset root (mostly gitignored)
│       │   └── FSC147/                # FSC-147 dataset files
│       │       └── sample/            # Small sample tracked in git
│       │
│       ├── GroundingDINO/             # Vendored GroundingDINO (installed as editable)
│       ├── output/                    # Training outputs (gitignored)
│       ├── output_transfer/           # Transfer learning outputs (gitignored)
│       ├── visualizations/            # Generated visualizations (gitignored)
│       ├── visualizations_test/       # Test visualizations (gitignored)
│       ├── wandb/                     # W&B run logs (gitignored)
│       └── notebooks/                 # Jupyter notebooks
│
├── demo/                              # Standalone demo deployment
│   ├── README.md
│   ├── requirements.txt
│   ├── demo_app_advanced.py           # Copy of main demo
│   ├── demo_inference.py              # Copy of inference module
│   ├── demo_pipeline_advanced.py      # Copy of pipeline
│   ├── demo_visualization.py          # Copy of visualization
│   ├── prompt_enhancer.py             # Copy of prompt enhancer
│   ├── models_mae_cross.py            # Copy of counting model
│   ├── models_crossvit.py             # Copy of cross-attention
│   ├── yolov8x-worldv2.pt            # YOLO weights (symlink or copy)
│   ├── data/                          # Symlink to data
│   ├── GroundingDINO/                 # Symlink to GroundingDINO
│   └── util/                          # Minimal util copy
│       ├── __init__.py
│       └── pos_embed.py
│
├── configs/                           # YAML training/eval configurations
│   ├── train_baseline.yaml
│   ├── train_finetune_dino_prompt.yaml
│   ├── train_finetune_yolo.yaml
│   └── test_baseline.yaml
│
├── scripts/                           # Shell helper scripts
│   ├── setup_env.sh                   # Environment setup
│   ├── download_data.sh               # Dataset download
│   ├── generate_exemplars.sh          # Exemplar generation (dino/yolo/all)
│   ├── run_evaluation.sh              # Full evaluation suite
│   ├── generate_per_image_csv.py      # Per-image CSV generation
│   ├── generate_plot_data.py          # Plot data generation
│   └── process_csv_to_dat.py          # CSV to dat conversion
│
├── experiments/                       # Archived experiment runs
│   ├── exp2/                          # VA-Count baseline
│   ├── exp3/                          # YOLO-World ablations
│   ├── exp4/                          # Mixed experiments
│   └── exp5/                          # Additional ablations
│
├── docs/                              # Documentation
│   ├── RESULTS.md                     # Numerical results & provenance
│   ├── report/                        # LaTeX report (main.tex)
│   ├── slide/                         # LaTeX presentation slides
│   └── references/                    # Reference materials
│
├── images/                            # GitHub repo images
├── env/                               # Environment files (empty)
└── tmp/                               # Temporary files
```

## Key Locations

| What | Where |
|------|-------|
| Core model | `code/source-code/models_mae_cross.py` |
| Training entry | `code/source-code/FSC_train.py` |
| Evaluation entry | `code/source-code/FSC_test.py` |
| Demo entry | `code/source-code/demo_app_advanced.py` |
| Dataset class | `code/source-code/util/FSC147.py` |
| Rich Prompt | `code/source-code/prompt_enhancer.py` |
| Config files | `configs/*.yaml` |
| Shell scripts | `scripts/` |
| Experiment logs | `experiments/exp{2,3,4,5}/` |
| LaTeX report | `docs/report/main.tex` |

## Naming Conventions

- **Model files:** `models_*.py` — different model architectures
- **Training/test:** `FSC_*.py` — scripts for FSC-147 dataset
- **Exemplar scripts:** `{grounding,yolo}_{pos,neg}*.py` — detector + polarity
- **Demo files:** `demo_*.py` — demo-related modules
- **Config files:** `{train,test}_*.yaml` — training or test configurations
- **Experiments:** `exp{N}/` — numbered experiment directories

## Code Duplication

The `demo/` directory is a **near-complete copy** of the relevant source files from `code/source-code/`. Files are duplicated (not symlinked in git):
- `demo_app_advanced.py`, `demo_inference.py`, `demo_pipeline_advanced.py`
- `demo_visualization.py`, `prompt_enhancer.py`
- `models_mae_cross.py`, `models_crossvit.py`
- `util/pos_embed.py`

Similarly, `experiments/exp2/util/` and `experiments/exp3/` contain copies of utility files.
