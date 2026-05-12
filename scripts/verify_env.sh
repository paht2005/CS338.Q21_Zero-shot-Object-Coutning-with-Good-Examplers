#!/usr/bin/env bash
# Smoke-test the conda environment on the lab server after server_setup.sh.
# Run from the repo root with the vacount conda env already activated.
# Usage: conda activate vacount && bash scripts/verify_env.sh
# Exit code: 0 if ALL checks pass, 1 otherwise.

set -uo pipefail

PASS=0
FAIL=0

check_pass() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
check_fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

echo "=============================================="
echo " VA-Count Environment Verification"
echo "=============================================="

# --- 1. Python import checks ---
echo ""
echo "[1/5] Python import checks..."

if python -c "import torch; print(f'  torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null; then
    check_pass "torch"
else
    check_fail "torch (import failed)"
fi

if python -c "import clip; print('  clip OK')" 2>/dev/null; then
    check_pass "clip"
else
    check_fail "clip (import failed)"
fi

if python -c "import ultralytics; print('  ultralytics OK')" 2>/dev/null; then
    check_pass "ultralytics"
else
    check_fail "ultralytics (import failed)"
fi

if python -c "import groundingdino; print('  groundingdino OK')" 2>/dev/null; then
    check_pass "groundingdino"
else
    check_fail "groundingdino (import failed)"
fi

if python -c "import google.generativeai; print('  google.generativeai OK')" 2>/dev/null; then
    check_pass "google.generativeai"
else
    check_fail "google.generativeai (import failed)"
fi

if python -c "import timm; print('  timm OK')" 2>/dev/null; then
    check_pass "timm"
else
    check_fail "timm (import failed)"
fi

# --- 2. GPU check ---
echo ""
echo "[2/5] GPU / CUDA check..."
if python -c "
import torch
assert torch.cuda.is_available(), 'No CUDA'
name = torch.cuda.get_device_name(0)
print(f'  CUDA device: {name}')
" 2>/dev/null; then
    check_pass "CUDA available"
else
    check_fail "CUDA not available (check nvidia-smi and torch CUDA version)"
fi

# --- 3. Data path checks ---
echo ""
echo "[3/5] NAS data path checks..."
NAS_PATHS=(
    "/mnt/mmlab2024nas/counting"
    "/mnt/mmlab2024nas/counting/fsc147/images_384_VarV2"
    "/mnt/mmlab2024nas/counting/fsc147/gt_density_map_adaptive_384_VarV2"
    "/mnt/mmlab2024nas/counting/fsc147/Train_Test_Val_FSC_147.json"
)
for path in "${NAS_PATHS[@]}"; do
    if [ -e "$path" ]; then
        check_pass "$path"
    else
        check_fail "MISSING: $path"
    fi
done

# --- 4. Checkpoint files check ---
echo ""
echo "[4/5] Checkpoint files (*.pth)..."
NAS_CKPT_DIR="/mnt/mmlab2024nas/counting/checkpoints"
LOCAL_CKPT_DIR="code/source-code/data"
FOUND_CKPTS=()

if [ -d "$NAS_CKPT_DIR" ]; then
    while IFS= read -r -d '' f; do
        FOUND_CKPTS+=("$f")
    done < <(find "$NAS_CKPT_DIR" -name "*.pth" -print0 2>/dev/null)
fi
if [ -d "$LOCAL_CKPT_DIR" ]; then
    while IFS= read -r -d '' f; do
        FOUND_CKPTS+=("$f")
    done < <(find "$LOCAL_CKPT_DIR" -name "*.pth" -print0 2>/dev/null)
fi

if [ ${#FOUND_CKPTS[@]} -gt 0 ]; then
    for ckpt in "${FOUND_CKPTS[@]}"; do
        echo "  Found: $ckpt"
    done
    check_pass "${#FOUND_CKPTS[@]} checkpoint(s) found"
else
    check_fail "No *.pth checkpoint files found in ${NAS_CKPT_DIR} or ${LOCAL_CKPT_DIR}"
fi

# --- 5. .env check ---
echo ""
echo "[5/5] .env / API key check..."
ENV_FILE="code/source-code/.env"
if [ -f "$ENV_FILE" ]; then
    if grep -q "GEMINI_API_KEY" "$ENV_FILE" 2>/dev/null; then
        check_pass ".env found with GEMINI_API_KEY entry"
    else
        check_fail ".env found but GEMINI_API_KEY not set"
    fi
else
    check_fail ".env missing at ${ENV_FILE} — copy from .env.example and fill in key"
fi

# --- Summary ---
TOTAL=$((PASS + FAIL))
echo ""
echo "=============================================="
echo " Summary: ${PASS}/${TOTAL} checks passed"
echo "=============================================="

if [ "$FAIL" -gt 0 ]; then
    echo " ${FAIL} check(s) FAILED — review output above"
    exit 1
else
    echo " All checks PASSED — server environment ready"
    exit 0
fi
