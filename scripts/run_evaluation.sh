#!/usr/bin/env bash
# Run evaluation on all model variants and print results
# Usage: bash scripts/run_evaluation.sh
# Assumes working directory is the repo root and conda env is activated.

set -euo pipefail

SRC="code/source-code"
DATA_DIR="${SRC}/data"

echo "=============================================="
echo " VA-Count Evaluation Suite"
echo "=============================================="

# --- Baseline (GroundingDINO, no prompt enhancement) ---
echo ""
echo "[1/3] Evaluating BASELINE (GroundingDINO)..."
cd "${SRC}"
python FSC_test.py \
    --data_split_file data/FSC147/Train_Test_Val_FSC_147.json \
    --im_dir data/FSC147/images_384_VarV2 \
    --gt_dir data/FSC147/gt_density_map_adaptive_384_VarV2 \
    --anno_file data/FSC147/annotation_FSC147_384.json \
    --output_dir output/test_baseline \
    --resume data/checkpoint_FSC.pth
cd ../..

# --- Fine-tuned with GroundingDINO + Rich Prompt ---
echo ""
echo "[2/3] Evaluating DINO + RICH PROMPT..."
cd "${SRC}"
python FSC_test.py \
    --data_split_file data/FSC147/Train_Test_Val_FSC_147.json \
    --im_dir data/FSC147/images_384_VarV2 \
    --gt_dir data/FSC147/gt_density_map_adaptive_384_VarV2 \
    --anno_file data/FSC147/annotation_FSC147_pos.json \
    --output_dir output/test_dino_prompt \
    --resume data/checkpoint__finetuning_dino_prompt.pth
cd ../..

# --- Fine-tuned with YOLO-World ---
echo ""
echo "[3/3] Evaluating YOLO-WORLD..."
cd "${SRC}"
python FSC_test.py \
    --data_split_file data/FSC147/Train_Test_Val_FSC_147.json \
    --im_dir data/FSC147/images_384_VarV2 \
    --gt_dir data/FSC147/gt_density_map_adaptive_384_VarV2 \
    --anno_file data/FSC147/annotation_FSC147_pos_yolo_prompt.json \
    --output_dir output/test_yolo \
    --resume data/checkpoint__finetuning_yolo.pth
cd ../..

echo ""
echo "=============================================="
echo " Evaluation complete. Check output/ for results."
echo "=============================================="
