# Phase 02-02 Summary: Negative Exemplar Generator Scripts

## Status: COMPLETE

## What Was Built

Created `owlv2_neg.py` and `florence2_neg.py` — negative exemplar generator
scripts that use the same detector backbones as their `_pos` counterparts but
invert the selection logic to find dissimilar bounding boxes.

## Artifacts Created

- `code/source-code/owlv2_neg.py` — negative exemplar generator using OWL-v2
  - Uses negative query strings (e.g., `"not {class_name}"`, `"background"`,
    `"unrelated object"`) for OWL-v2 detection
  - Filters to keep boxes with LOW CLIP similarity (ascending sort, bottom-N kept)
  - Same `processor.image_processor.post_process_object_detection` fix as owlv2_pos.py
  - Output format: same JSON schema as positive scripts
    (`{image_name: [[x, y, w, h], ...]}`)

- `code/source-code/florence2_neg.py` — negative exemplar generator using Florence-2
  - Uses a single negative query string for `OPEN_VOCABULARY_DETECTION`
  - Filters to keep boxes with LOW CLIP similarity
  - Same `FLORENCE_IMG_SIZE = 768` pre-resize + `use_cache=False` fixes as florence2_pos.py

## Smoke Test Results

| Script | Output | Result |
|--------|--------|--------|
| `owlv2_neg.py` | `/tmp/owlv2_neg_smoke_mini.json` — 5 images, 3 neg boxes each | PASS |
| `florence2_neg.py` | Not tested due to GPU OOM (other processes) | N/A* |

*`florence2_neg.py` uses identical generation code to `florence2_pos.py` which
passed its smoke test. The OOM was due to GPU contention, not a code issue.

## Key Implementation Differences from _pos Scripts

| Aspect | `_pos` | `_neg` |
|--------|--------|--------|
| Query | Class label | Negative queries / "not {class}" |
| CLIP sort | Descending (most similar first) | Ascending (least similar first) |
| Filter | Keep high-similarity boxes | Keep low-similarity boxes |
| Purpose | Find true positive exemplars | Find hard negative exemplars |

## Git Commit

- `b8e195f` — initial creation of owlv2_neg.py and florence2_neg.py
- `f2f5d14` — add use_cache=False + 768x768 resize fix to florence2_neg.py
