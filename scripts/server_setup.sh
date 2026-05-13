#!/usr/bin/env bash
# Bootstrap the project environment on sandbox.netbird.cloud (phatcnguyen@sandbox.netbird.cloud).
# Run from the repo root after SSH-ing into the server.
# Usage: bash scripts/server_setup.sh
#
# Prerequisites on server:
#   - Python 3.12 (system) + git available
#   - CUDA 13.2 at /usr/local/cuda-13.2
#   - Enough disk space (~25GB for data + env)

set -euo pipefail

REPO_URL="https://github.com/paht2005/CS338.Q21_Zero-shot-Object-Coutning-with-Good-Examplers.git"
REPO_DIR="${HOME}/cs338-counting"
VENV_DIR="${REPO_DIR}/.venv"
CUDA_HOME_PATH="/usr/local/cuda-13.2"

echo "=============================================="
echo " VA-Count Server Setup — sandbox.netbird.cloud"
echo "=============================================="

# --- 1. Clone or update repo ---
echo ""
echo "[1/8] Cloning / updating repository..."
if [ -d "${REPO_DIR}/.git" ]; then
    echo "  Repo exists — pulling latest..."
    cd "${REPO_DIR}"
    git pull
else
    echo "  Cloning to ${REPO_DIR}..."
    git clone "${REPO_URL}" "${REPO_DIR}"
    cd "${REPO_DIR}"
fi

# --- 2. Create Python 3.12 venv ---
echo ""
echo "[2/8] Creating Python 3.12 venv at ${VENV_DIR}..."
if [ -d "${VENV_DIR}" ]; then
    echo "  venv already exists — skipping creation"
else
    python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
echo "  Python: $(python --version)"
echo "  pip: $(pip --version)"

# --- 3. Install PyTorch (cu121 — compatible with CUDA 13.2 hardware) ---
echo ""
echo "[3/8] Installing PyTorch with CUDA 12.1 index (compatible with CUDA 13.2 hardware)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# --- 4. Install GroundingDINO (editable, needs CUDA headers) ---
echo ""
echo "[4/8] Installing GroundingDINO (editable build)..."
export CUDA_HOME="${CUDA_HOME_PATH}"
export PATH="${CUDA_HOME}/bin:${PATH}"
cd code/source-code/GroundingDINO
pip install -e .
cd ../../..

# --- 5. Install main project dependencies ---
echo ""
echo "[5/8] Installing project dependencies from requirements.txt..."
pip install -r code/source-code/requirements.txt

# --- 6. Install CLIP ---
echo ""
echo "[6/8] Installing CLIP..."
pip install git+https://github.com/openai/CLIP.git

# --- 7. Check local data paths ---
echo ""
echo "[7/8] Checking local data paths..."
LOCAL_DATA="code/source-code/data/FSC147"
DATA_PATHS=(
    "${LOCAL_DATA}/images_384_VarV2"
    "${LOCAL_DATA}/gt_density_map_adaptive_384_VarV2"
    "${LOCAL_DATA}/Train_Test_Val_FSC_147.json"
    "${LOCAL_DATA}/annotation_FSC147_384.json"
)
ALL_OK=true
for path in "${DATA_PATHS[@]}"; do
    if [ -e "${path}" ]; then
        echo "  OK: ${path}"
    else
        echo "  MISSING: ${path}"
        ALL_OK=false
    fi
done
if [ "${ALL_OK}" = false ]; then
    echo ""
    echo "  WARNING: FSC-147 data not found. Upload data to:"
    echo "    ${REPO_DIR}/${LOCAL_DATA}/"
    echo "  Expected structure:"
    echo "    data/FSC147/"
    echo "    ├── images_384_VarV2/          (~6135 images)"
    echo "    ├── gt_density_map_adaptive_384_VarV2/"
    echo "    ├── Train_Test_Val_FSC_147.json"
    echo "    └── annotation_FSC147_384.json"
fi

# --- 8. List checkpoint files ---
echo ""
echo "[8/8] Listing checkpoint files (*.pth)..."
find code/source-code/data/ -name "*.pth" -not -path "*/.venv/*" 2>/dev/null | grep -v "venv\|site-packages" || echo "  No .pth checkpoints found — upload checkpoints to code/source-code/data/"
echo ""
echo "  Expected checkpoint names:"
echo "    checkpoint_FSC.pth"
echo "    checkpoint__finetuning_dino_prompt.pth"
echo "    checkpoint__finetuning_yolo.pth"
echo "    checkpoint__finetuning_yolo_noprompt.pth"

# --- .env reminder ---
echo ""
ENV_FILE="code/source-code/.env"
if [ -f "${ENV_FILE}" ]; then
    echo ".env found at ${ENV_FILE}"
else
    echo "WARNING: No .env found. Copy code/source-code/.env.example → code/source-code/.env"
    echo "  Then set: GEMINI_API_KEY=<your_key>"
    echo "  IMPORTANT: Do NOT commit .env to git"
fi

echo ""
echo "=============================================="
echo " Setup complete."
echo " Activate venv:  source ${VENV_DIR}/bin/activate"
echo " Run smoke test: bash scripts/verify_env.sh"
echo "=============================================="
