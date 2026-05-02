#!/usr/bin/env bash
# Download FSC147 dataset and model checkpoints
# Usage: bash scripts/download_data.sh
#
# This script assumes you have the OneDrive/Google Drive links.
# Replace the placeholder URLs with real download links.

set -euo pipefail

DATA_DIR="code/source-code/data"

echo "=== Downloading FSC147 Dataset ==="
echo ""
echo "The FSC147 dataset must be downloaded manually:"
echo "  1. Visit: https://github.com/cvlab-stonybrook/LearningToCountEverything"
echo "  2. Download images_384_VarV2.zip"
echo "  3. Download gt_density_map_adaptive_384_VarV2.zip"
echo "  4. Download annotation files"
echo "  5. Extract to: ${DATA_DIR}/FSC147/"
echo ""
echo "Expected structure:"
echo "  ${DATA_DIR}/FSC147/"
echo "  ├── images_384_VarV2/"
echo "  ├── gt_density_map_adaptive_384_VarV2/"
echo "  ├── annotation_FSC147_384.json"
echo "  ├── Train_Test_Val_FSC_147.json"
echo "  └── ImageClasses_FSC147.txt"
echo ""

echo "=== Model Checkpoints ==="
echo ""
echo "Download checkpoints and place them in: ${DATA_DIR}/"
echo "  - checkpoint_FSC.pth                          (baseline VA-Count)"
echo "  - checkpoint__finetuning_dino_prompt.pth      (GroundingDINO + Rich Prompt)"
echo "  - checkpoint__finetuning_yolo.pth             (YOLO-World)"
echo "  - checkpoint__finetuning_yolo_noprompt.pth    (YOLO-World, no prompt)"
echo ""
echo "NOTE: These files are >100MB and stored on OneDrive."
echo "Link: (paste OneDrive link in README.md)"
