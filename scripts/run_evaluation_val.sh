#!/usr/bin/env bash
# Run val-split evaluation for all 4 pipeline configurations on sandbox.netbird.cloud.
# Run from the repo root with .venv activated.
# Usage: source .venv/bin/activate && bash scripts/run_evaluation_val.sh

set -uo pipefail

# --- Paths (all absolute from repo root) ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_BASE="${DATA_BASE:-${REPO_ROOT}/code/source-code/data}"
FSC_DIR="${DATA_BASE}/FSC147"
CKPT_DIR="${DATA_BASE}"
SRC="${REPO_ROOT}/code/source-code"
OUT_BASE="${REPO_ROOT}/experiments/server-val-baseline"

mkdir -p "${OUT_BASE}"

echo "=============================================="
echo " VA-Count Val-Split Evaluation"
echo " Data: ${FSC_DIR}"
echo " Output: ${OUT_BASE}"
echo "=============================================="

# Helper: run one config; skips if files missing
# Args: NAME  ANNO_FILENAME  CKPT_PATH
run_config() {
    local NAME="$1"
    local ANNO_FILE="$2"  # just filename, relative to FSC_DIR
    local CKPT="$3"

    echo ""
    echo "--- Config: ${NAME} ---"

    if [ ! -f "${FSC_DIR}/${ANNO_FILE}" ]; then
        echo "  SKIPPED: annotation file not found: ${FSC_DIR}/${ANNO_FILE}"
        return
    fi
    if [ ! -f "${CKPT}" ]; then
        echo "  SKIPPED: checkpoint not found: ${CKPT}"
        return
    fi

    mkdir -p "${SRC}/output/val_${NAME}"
    cd "${SRC}"
    python FSC_test.py \
        --data_path       "${FSC_DIR}" \
        --anno_file       "${ANNO_FILE}" \
        --data_split_file "Train_Test_Val_FSC_147.json" \
        --im_dir          "images_384_VarV2" \
        --output_dir      "output/val_${NAME}" \
        --resume          "${CKPT}" \
        --split val \
        --external \
        2>&1 | tee "${OUT_BASE}/${NAME}.log"
    cd "${REPO_ROOT}"
    echo "  Done: experiments/server-val-baseline/${NAME}.log"
}

# Config 1: VA-Count Baseline (GroundingDINO, raw prompt)
run_config "baseline" \
    "annotation_FSC147_384.json" \
    "${CKPT_DIR}/checkpoint_FSC.pth"

# Config 2: VA-Count + Rich Prompt (GroundingDINO + Gemini + CLIP)
run_config "dino_rich" \
    "annotation_FSC147_pos.json" \
    "${CKPT_DIR}/checkpoint__finetuning_dino_prompt.pth"

# Config 3: VA-Count + YOLO-World (no Rich Prompt)
run_config "yolo_norich" \
    "annotation_FSC147_pos_yolo_noprompt.json" \
    "${CKPT_DIR}/checkpoint__finetuning_yolo_noprompt.pth"

# Config 4: VA-Count + YOLO-World + Rich Prompt
run_config "yolo_rich" \
    "annotation_FSC147_pos_yolo_prompt.json" \
    "${CKPT_DIR}/checkpoint__finetuning_yolo.pth"

# --- Summary ---
echo ""
echo "=============================================="
echo " RESULTS SUMMARY"
echo "=============================================="
echo " Grep MAE/RMSE from logs:"
grep -rE "MAE|RMSE" "${OUT_BASE}"/*.log 2>/dev/null || echo "  No results yet"
