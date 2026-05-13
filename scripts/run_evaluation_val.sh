#!/usr/bin/env bash
# Run val-split evaluation for all 4 pipeline configurations on sandbox.netbird.cloud.
# Run from the repo root with .venv activated.
# Usage: source .venv/bin/activate && bash scripts/run_evaluation_val.sh
#
# Override data path with env var:
#   DATA_BASE=/path/to/data bash scripts/run_evaluation_val.sh

set -uo pipefail

# --- Paths ---
DATA_BASE="${DATA_BASE:-code/source-code/data}"
FSC_DIR="${DATA_BASE}/FSC147"
CKPT_DIR="${DATA_BASE}"
SRC="code/source-code"
OUT_BASE="experiments/server-val-baseline"

mkdir -p "${OUT_BASE}"

echo "=============================================="
echo " VA-Count Val-Split Evaluation"
echo " Data: ${FSC_DIR}"
echo " Output: ${OUT_BASE}"
echo "=============================================="

# Helper: run one config; skips if files missing
run_config() {
    local NAME="$1"
    local ANNO="$2"
    local CKPT="$3"

    echo ""
    echo "--- Config: ${NAME} ---"

    if [ ! -f "${ANNO}" ]; then
        echo "  SKIPPED: annotation file not found: ${ANNO}"
        return
    fi
    if [ ! -f "${CKPT}" ]; then
        echo "  SKIPPED: checkpoint not found: ${CKPT}"
        return
    fi

    mkdir -p "${OUT_BASE}"
    cd "${SRC}"
    python FSC_test.py \
        --data_split_file "data/FSC147/Train_Test_Val_FSC_147.json" \
        --im_dir          "data/FSC147/images_384_VarV2" \
        --gt_dir          "data/FSC147/gt_density_map_adaptive_384_VarV2" \
        --anno_file       "${ANNO}" \
        --output_dir      "output/val_${NAME}" \
        --resume          "${CKPT}" \
        --split val \
        --external \
        2>&1 | tee "../../${OUT_BASE}/${NAME}.log"
    cd ../..
    echo "  Done: ${OUT_BASE}/${NAME}.log"
}

# Config 1: VA-Count Baseline (GroundingDINO, raw prompt)
run_config "baseline" \
    "${FSC_DIR}/annotation_FSC147_384.json" \
    "${CKPT_DIR}/checkpoint_FSC.pth"

# Config 2: VA-Count + Rich Prompt (GroundingDINO + Gemini + CLIP)
run_config "dino_rich" \
    "${FSC_DIR}/annotation_FSC147_pos.json" \
    "${CKPT_DIR}/checkpoint__finetuning_dino_prompt.pth"

# Config 3: VA-Count + YOLO-World (no Rich Prompt)
run_config "yolo_norich" \
    "${FSC_DIR}/annotation_FSC147_pos_yolo_noprompt.json" \
    "${CKPT_DIR}/checkpoint__finetuning_yolo_noprompt.pth"

# Config 4: VA-Count + YOLO-World + Rich Prompt
run_config "yolo_rich" \
    "${FSC_DIR}/annotation_FSC147_pos_yolo_prompt.json" \
    "${CKPT_DIR}/checkpoint__finetuning_yolo.pth"

# --- Summary ---
echo ""
echo "=============================================="
echo " RESULTS SUMMARY"
echo "=============================================="
echo " Grep MAE/RMSE from logs:"
grep -rE "MAE|RMSE|mae|rmse" "${OUT_BASE}"/*.log 2>/dev/null || echo "  No results yet (logs may be empty or configs skipped)"
