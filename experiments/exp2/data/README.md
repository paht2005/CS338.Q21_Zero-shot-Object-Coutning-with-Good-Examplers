# VA-Count: Zero-shot Object Counting with Good Exemplars
# This project re-implements the VA-Count paper with new improvements


## Table of Contents
- [News](#news)
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Model Checkpoints](#model-checkpoints)
- [Quick Start](#quick-start)
- [Training](#training)
- [Inference](#inference)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Project History

> This README is an archived snapshot kept inside `experiments/exp2/data/` for
> reproducibility of the original `exp2` run. For the up-to-date CS338.Q21
> instructions, use `code/source-code/README.md` at the repository root.

- ECCV 2024: Original VA-Count paper published.
- CS338.Q21 (2026): Re-implementation of VA-Count from the paper, extended with
  Rich Prompt and YOLO-World, plus the LaTeX report under
  `docs/report/Report.pdf`.

## Overview
VA-Count is a zero-shot object counting method that leverages good exemplars for accurate counting. The model combines:
- Vision Transformer backbone (MAE pretrained)
- Grounding DINO for exemplar detection
- Binary classifier for single/multiple object detection
- Cross-attention mechanism for feature matching

## Project Structure

```
VA-Count/
├── data/                           # Dataset directory
│   ├── checkpoint.pth              # Main model checkpoint
│   ├── checkpoint_FSC.pth          # FSC147 trained checkpoint
│   ├── checkpoint__finetuning_dino_prompt.pth    # DINO with prompt finetuned
│   ├── checkpoint__finetuning_yolo.pth           # YOLO with prompt finetuned
│   ├── checkpoint__finetuning_yolo_noprompt.pth  # YOLO without prompt finetuned
│   ├── FSC147/
│   │   ├── images_384_VarV2/      # Resized images (384x384)
│   │   ├── gt_density_map_adaptive_384_VarV2/  # Ground truth density maps
│   │   ├── annotation_FSC147_384.json          # Original annotations
│   │   ├── annotation_FSC147_pos.json          # Positive exemplars (DINO without prompt)
│   │   ├── annotation_FSC147_pos_prompt.json   # Positive exemplars (DINO with prompt)
│   │   ├── annotation_FSC147_neg.json          # Negative exemplars
│   │   ├── annotation_FSC147_neg_prompt.json   # Negative exemplars (DINO with prompt)
│   │   ├── annotation_FSC147_neg_yolo.json     # YOLO negative annotations
│   │   ├── annotation_FSC147_neg_yolo_prompt.json  # YOLO negative with prompts
│   │   ├── Train_Test_Val_FSC_147.json         # Official train/val/test split
│   │   ├── ImageClasses_FSC147.txt             # Image class labels
│   │   ├── ImageClasses_FSC147_detailed_v6_pos.txt # Rich prompt for positive
│   │   ├── ImageClasses_FSC147_detailed_v6_neg.txt # Rich prompt for negative
│   │   ├── train.txt               # Training image list
│   │   ├── val.txt                 # Validation image list
│   │   └── test.txt                # Test image list
│   ├── CARPK/                      # CARPK dataset (optional)
│   └── out/                        # Output directory
│       ├── classify/               # Classifier checkpoints
│       ├── results_base/           # Test results
│       └── pre_4_dir/              # Pretrain checkpoints
├── GroundingDINO/                  # Grounding DINO submodule
│   ├── groundingdino/
│   ├── weights/
│   │   └── groundingdino_swint_ogc.pth
│   └── ...
├── util/                           # Utility functions
│   ├── FSC147.py                  # Dataset loader
│   ├── misc.py                    # Miscellaneous utilities
│   └── ...
├── models_crossvit.py             # Cross-ViT model
├── models_mae_cross.py            # MAE with cross-attention
├── models_mae_noct.py             # MAE without counting token
├── FSC_pretrain.py                # Pretraining script
├── FSC_train.py                   # Training script
├── FSC_test.py                    # Testing script
├── biclassify.py                  # Binary classifier training
├── datasetmake.py                 # Dataset preparation
├── grounding_pos.py               # Generate positive exemplars
├── grounding_neg.py               # Generate negative exemplars
├── yolo_pos_withPrompt.py         # YOLO with prompts (positive)
├── yolo_neg.py                    # YOLO for negative examples
├── yolo_pos_withoutPrompt.py      # YOLO without prompts
├── demo_app_advanced.py           # Advanced demo application
├── demo_inference.py              # Basic inference demo
├── demo_pipeline_advanced.py      # Advanced pipeline demo
├── demo_visualization.py          # Visualization demo
├── inference_official.py          # Official inference script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Environment Setup

### Prerequisites
- Python 3.8+
- CUDA 11.3+ (for GPU support)
- PyTorch 1.12+

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/paht2005/CS338.Q21_Zero-shot-Object-Coutning-with-Good-Examplers.git
cd CS338.Q21_Zero-shot-Object-Coutning-with-Good-Examplers
```

2. **Create conda environment**
```bash
conda create -n vacount python=3.12
conda activate vacount
```

3. **Install PyTorch** (example for CUDA 11.3)
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

4. **Install Grounding DINO**
```bash
cd GroundingDINO
pip install -e .
cd ..
```

5. **Install other dependencies**
```bash
pip install -r requirements.txt
```

6. **Download Grounding DINO weights**
```bash
mkdir -p GroundingDINO/weights
cd GroundingDINO/weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../..
```

## Dataset Preparation

### FSC147 Dataset

1. **Download FSC147**
   - Download from [FSC147 GitHub](https://github.com/cvlab-stonybrook/LearningToCountEverything)
   - Extract to `./data/FSC147/`

2. **Prepare data splits**
```bash
# Create train/val/test split files
python -c "
import json
from pathlib import Path

json_file = './data/FSC147/Train_Test_Val_FSC_147.json'
with open(json_file, 'r') as f:
    data = json.load(f)

for split in ['train', 'val', 'test']:
    with open(f'./data/FSC147/{split}.txt', 'w') as f:
        for item in data[split]:
            f.write(f'{item}\n')
"
```

### Expected Directory Structure
```
./data/FSC147/
├── images_384_VarV2/
│   ├── 2.jpg
│   ├── 3.jpg
│   └── ...
├── gt_density_map_adaptive_384_VarV2/
│   ├── 2.npy
│   ├── 3.npy
│   └── ...
├── annotation_FSC147_384.json
├── Train_Test_Val_FSC_147.json
├── ImageClasses_FSC147.txt
├── train.txt
├── val.txt
└── test.txt
```

## Model Checkpoints

### Download Pretrained Models (Original paper)

1. **Main counting model** (required for inference)
   - Download from [Baidu Disk](https://pan.baidu.com/s/11sbdDYLDfTOIPx5pZvBpmw?pwd=paeh) (Password: `paeh`)
   - Save to `./data/checkpoint_FSC.pth`

2. **Binary classifier** (optional, for exemplar selection)
   - Download from [Baidu Disk](https://pan.baidu.com/s/1fOF0giI3yQpvGTiNFUI7cQ?pwd=psum) (Password: `psum`)
   - Save to `./data/out/classify/`

3. **MAE pretrained backbone** (required for training)
   - Download from official MAE repository
   - Save to `./weights/mae_pretrain_vit_base_full.pth`

## Quick Start

### Advanced Demo with Visualization

```bash
# Run advanced demo with visual outputs
python demo_app_advanced.py \
    --resume ./data/checkpoint_FSC.pth \
    --data_path ./data/FSC147 \
    --output_dir ./demo_outputs \
    --visualize
```

### Official Testing

```bash
# Test on FSC147 test set
python FSC_test.py \
    --output_dir ./data/out/results_base \
    --resume ./data/checkpoint_FSC.pth \
    --data_path ./data/FSC147 \
    --split test
```

## Training

### Step 1: Prepare Binary Classifier (Optional)

```bash
# Generate dataset for classifier
python datasetmake.py --data_path ./data/FSC147

# Train binary classifier
python biclassify.py \
    --data_path ./data/FSC147 \
    --output_dir ./data/out/classify \
    --epochs 100
```

### Step 2: Generate Exemplars

```bash
# Generate positive exemplars using Grounding DINO
python grounding_pos.py --root_path ./data/FSC147/

# Generate negative exemplars
python grounding_neg.py --root_path ./data/FSC147/
```

Alternative: Use YOLO for exemplar generation
```bash
# With prompts
python yolo_pos_withPrompt.py --root_path ./data/FSC147/

# Without prompts
python yolo_pos_withoutPrompt.py --root_path ./data/FSC147/
```

### Step 3: Pretraining (Optional)

```bash
# Pretrain the model
python FSC_pretrain.py \
    --data_path ./data/FSC147 \
    --output_dir ./data/out/pre_4_dir \
    --resume ./weights/mae_pretrain_vit_base_full.pth \
    --epochs 300 \
    --batch_size 8 \
    --lr 1e-4
```

### Step 4: Fine-tuning

```bash
# Fine-tune with positive exemplars
python FSC_train.py \
    --data_path ./data/FSC147 \
    --anno_file annotation_FSC147_pos.json \
    --output_dir ./data/out/finetune_pos \
    --resume ./data/out/pre_4_dir/checkpoint-latest.pth \
    --epochs 500 \
    --batch_size 8 \
    --lr 1e-5
```

## Inference


### Batch Inference

```bash
python FSC_test.py
```

### Streamlit demo

```bash
streamlit run demo_app_advanced.py
```



## Citation

```bibtex
@inproceedings{zhu2024zero,
  title={Zero-shot Object Counting with Good Exemplars},
  author={Zhu, Huilin and Yuan, Jingling and Yang, Zhengwei and Guo, Yu and Wang, Zheng and Zhong, Xian and He, Shengfeng},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2024}
}
```

## Acknowledgement

This project is based on:
- [CounTR](https://github.com/Verg-Avesta/CounTR) - Base counting architecture
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) - Exemplar detection
- [MAE](https://github.com/facebookresearch/mae) - Vision Transformer backbone

We are very grateful for these excellent works!

## Contact

For questions about this CS338.Q21 project, please contact the team leader
**Nguyen Cong Phat — `23521143@gm.uit.edu.vn`**. For questions about the
original VA-Count paper, please refer to the upstream authors.

## License

This project is licensed under the MIT License — see [LICENSE.txt](../../../LICENSE.txt)
at the repository root for details.