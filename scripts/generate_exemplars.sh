#!/usr/bin/env bash
# Generate exemplars using both GroundingDINO and YOLO-World pipelines
# Usage: bash scripts/generate_exemplars.sh [dino|yolo|all]

set -euo pipefail

SRC="code/source-code"
MODE="${1:-all}"

cd "${SRC}"

if [[ "${MODE}" == "dino" || "${MODE}" == "all" ]]; then
    echo "=== Generating POSITIVE exemplars with GroundingDINO ==="
    python grounding_pos.py

    echo "=== Generating NEGATIVE exemplars with GroundingDINO ==="
    python grounding_neg.py
fi

if [[ "${MODE}" == "yolo" || "${MODE}" == "all" ]]; then
    echo "=== Generating POSITIVE exemplars with YOLO-World (with prompt) ==="
    python yolo_pos_withPrompt.py

    echo "=== Generating POSITIVE exemplars with YOLO-World (without prompt) ==="
    python yolo_pos_withoutPrompt.py

    echo "=== Generating NEGATIVE exemplars with YOLO-World ==="
    python yolo_neg.py
fi

echo ""
echo "=== Exemplar generation complete ==="
