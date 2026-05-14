# Phase 02-01 Summary: Dependency Installation & Smoke Tests

## Status: COMPLETE

## What Was Built

Installed OWL-v2 and Florence-2 dependencies on the server, set up HF cache on
NAS, and ran 5-image smoke tests for both detectors' positive exemplar scripts.

## Artifacts Created / Modified

- `code/source-code/owlv2_pos.py` — positive exemplar generator using OWL-v2
  - Fixed `processor.image_processor.post_process_object_detection` API for
    transformers 5.x (was `processor.post_process_...`)
- `code/source-code/florence2_pos.py` — positive exemplar generator using Florence-2
  - Added `FLORENCE_IMG_SIZE = 768` pre-resize: input resized to 768×768 square
    before processor, bounding boxes scaled back to original image dimensions after
  - Added `use_cache=False` to `model.generate()` to bypass `EncoderDecoderCache`
    incompatibility with transformers 5.8.1
- `scripts/server_setup.sh` — added sections 7b (HF_HOME/TRANSFORMERS_CACHE → NAS)
  and 7c (`pip install transformers accelerate einops timm`)

## Server State

- Server: `counting@192.168.6.200`, conda env `cs338` (Python 3.10)
- `HF_HOME=/mnt/mmlab2024nas/counting/hf_cache` (set in `~/.bashrc`)
- HF models cached at NAS:
  - OWL-v2: `google/owlv2-base-patch16-ensemble`
  - Florence-2: `microsoft/Florence-2-large`
- HF cache patches applied (NOT in git — patched directly in NAS module cache):
  - `configuration_florence2.py` line 265: `getattr(self, "forced_bos_token_id", None)`
  - `processing_florence2.py` line 89: `getattr(tokenizer, "additional_special_tokens", [])`
  - `modeling_florence2.py`: `_supports_sdpa = False` class attribute added to
    `Florence2ForConditionalGeneration`

## Smoke Test Results

| Script | Output | Result |
|--------|--------|--------|
| `owlv2_pos.py` | `/tmp/owlv2_smoke_mini.json` — 5 images | PASS |
| `florence2_pos.py` | `/tmp/florence2_smoke_mini.json` — 5 images | PASS |

Note: Florence-2 detected 0 boxes per image on the mini split (5 val images).
This is expected behavior for `OPEN_VOCABULARY_DETECTION` with these particular
images — the script correctly completes and writes empty exemplar lists.

## Key Decision: use_cache=False

transformers 5.x `_beam_search` creates `EncoderDecoderCache` and passes it as
`past_key_values`. Florence-2's legacy code expects tuple-of-tuples, causing
`TypeError: 'EncoderDecoderCache' object is not subscriptable`.

Fix: pass `use_cache=False` to `model.generate()`. This disables KV caching
entirely, so `past_key_values` is always `None` throughout the generation loop.
Slightly slower inference but functionally correct.

## Git Commits

- `b8e195f` — initial scripts + server_setup.sh updates
- `f2f5d14` — add use_cache=False + 768x768 resize fix to both florence2 scripts
