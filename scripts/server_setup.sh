#!/usr/bin/env bash
# Bootstrap the conda environment on the lab server (counting@192.168.6.200).
# Run from the repo root after SSH-ing into the server.
# Usage: bash scripts/server_setup.sh [env_name]
#   env_name: optional conda environment name (default: vacount)

set -euo pipefail

ENV_NAME="${1:-vacount}"
PYTHON_VERSION="3.10"

echo "=============================================="
echo " VA-Count Server Setup"
echo " Env: ${ENV_NAME}  Python: ${PYTHON_VERSION}"
echo "=============================================="

# --- 1. Create conda environment ---
echo ""
echo "[1/9] Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

# --- 2. Activate environment ---
echo ""
echo "[2/9] Activating environment..."
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# --- 3. Install PyTorch with CUDA ---
echo ""
echo "[3/9] Installing PyTorch (cu118)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# --- 4. Install GroundingDINO (editable) ---
echo ""
echo "[4/9] Installing GroundingDINO (editable)..."
cd code/source-code/GroundingDINO
pip install -e .
cd ../../..

# --- 5. Install main project dependencies ---
echo ""
echo "[5/9] Installing project dependencies from requirements.txt..."
pip install -r code/source-code/requirements.txt

# --- 6. Install CLIP ---
echo ""
echo "[6/9] Installing CLIP..."
pip install git+https://github.com/openai/CLIP.git

# --- 7. Verify NAS data paths ---
echo ""
echo "[7/9] Checking NAS data paths..."
NAS_PATHS=(
    "/mnt/mmlab2024nas/counting"
    "/mnt/mmlab2024nas/counting/fsc147/images_384_VarV2"
    "/mnt/mmlab2024nas/counting/fsc147/gt_density_map_adaptive_384_VarV2"
    "/mnt/mmlab2024nas/counting/fsc147/Train_Test_Val_FSC_147.json"
)
ALL_PATHS_OK=true
for path in "${NAS_PATHS[@]}"; do
    if [ -e "$path" ]; then
        echo "  OK: $path"
    else
        echo "  MISSING: $path"
        ALL_PATHS_OK=false
    fi
done
if [ "$ALL_PATHS_OK" = false ]; then
    echo "WARNING: Some NAS paths are missing. Check NFS mount."
fi

# --- 8. List checkpoint files ---
echo ""
echo "[8/9] Listing checkpoint files (*.pth)..."
echo "  Searching /mnt/mmlab2024nas/counting/..."
find /mnt/mmlab2024nas/counting/ -name "*.pth" 2>/dev/null | head -20 || echo "  (none found or path inaccessible)"
echo "  Searching code/source-code/data/..."
find code/source-code/data/ -name "*.pth" 2>/dev/null | head -20 || echo "  (none found)"

# --- 9. Print .env reminder ---
echo ""
echo "[9/9] Environment configuration reminder..."
ENV_FILE="code/source-code/.env"
if [ -f "$ENV_FILE" ]; then
    echo "  .env found at $ENV_FILE"
else
    echo "  WARNING: No .env file found at $ENV_FILE"
    echo "  Copy code/source-code/.env.example to code/source-code/.env"
    echo "  Then fill in your GEMINI_API_KEY"
    echo "  IMPORTANT: Do NOT commit .env to git"
fi

echo ""
echo "=============================================="
echo " Setup complete."
echo " Run: bash scripts/verify_env.sh"
echo "=============================================="
