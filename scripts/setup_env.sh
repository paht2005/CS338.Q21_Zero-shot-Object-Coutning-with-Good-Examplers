#!/usr/bin/env bash
# Setup environment for VA-Count project
# Usage: bash scripts/setup_env.sh

set -euo pipefail

ENV_NAME="${1:-vacount}"
PYTHON_VERSION="3.12"

echo "=== Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION}) ==="
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "=== Installing GroundingDINO ==="
cd code/source-code/GroundingDINO
pip install -e .
cd ../../..

echo "=== Installing Python dependencies ==="
pip install -r code/source-code/requirements.txt

echo "=== Setup complete ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo ""
echo "Next steps:"
echo "  1. Copy code/source-code/.env.example to code/source-code/.env"
echo "  2. Fill in your GEMINI_API_KEY in .env"
echo "  3. Download FSC147 dataset to code/source-code/data/FSC147/"
echo "  4. Download model checkpoints to code/source-code/data/"
