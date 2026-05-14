---
plan: 03-01
phase: 03-exemplar-generation-for-additional-detectors
status: complete
commit: 9e9ea92
---

# Summary: Plan 03-01 — generate_prompt_text.py + generate_exemplars.sh

## What was built

Created `code/source-code/generate_prompt_text.py`, a bulk Gemini rich-prompt text generator, and extended `scripts/generate_exemplars.sh` with full Phase 3 generation modes.

## Key files

### Created
- `code/source-code/generate_prompt_text.py` — Bulk Gemini annotator for FSC147 val split. Reads `GEMINI_API_KEY` from env only, fully resumable (checkpoint every 100 images), produces `annotation_FSC147_pos_prompt_text.json`.

### Modified
- `scripts/generate_exemplars.sh` — Extended with 6 new modes: `owlv2`, `florence2`, `owlv2-prompt`, `florence2-prompt`, `all-new`, plus updated `all`. Passes `--prompt` flag to owlv2/florence2 scripts for DET-04 runs. Validates that the prompt-text JSON prerequisite exists before `--prompt` runs.

## Decisions made

- `generate_prompt_text.py` imports `enhance_prompt_with_gemini` directly from `prompt_enhancer.py` — no duplication of Gemini client init
- Checkpoint write every 100 images (safe to interrupt and resume)
- `--delay 0.5` default for Gemini rate limiting
- `generate_exemplars.sh` uses env-var overrides (`TEXT_FILE`, `DATASET`, `DATA_ROOT`, `SPLIT`) so individual paths can be customized without editing the script

## Self-check

- [x] `GEMINI_API_KEY` read from `os.environ` only, never hardcoded
- [x] `enhance_prompt_with_gemini` imported from `prompt_enhancer.py`
- [x] Resume logic present (`result_map` loaded from existing output file)
- [x] Checkpoint write every 100 images
- [x] `generate_exemplars.sh` syntax valid (`bash -n` passes)
- [x] All 8 Phase 3 commands documented in script header
- [x] `--prompt` prerequisite guard in `owlv2-prompt` and `florence2-prompt` modes
