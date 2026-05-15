# Plan 03-02 Summary: Run 4 Raw Exemplar Generation Jobs

**Phase:** 03-exemplar-generation-for-additional-detectors
**Plan:** 02
**Completed:** 2026-05-15

## What Was Done

Ran 4 exemplar generation jobs on `counting@192.168.6.200` for the val split (1286 images) of FSC-147:

| Script | Output file | Size | Finished |
|--------|------------|------|---------|
| `owlv2_pos.py` | `annotation_FSC147_pos_owlv2.json` | 82K | ~01:05 |
| `owlv2_neg.py` | `annotation_FSC147_neg_owlv2.json` | 178K | ~01:09 |
| `florence2_pos.py` | `annotation_FSC147_pos_florence2.json` | 82K | 05:30 |
| `florence2_neg.py` | `annotation_FSC147_neg_florence2.json` | 82K | 09:56 |

All 4 files downloaded locally to `code/source-code/data/FSC147/` (gitignored — large data files).

## Issues Encountered & Resolved

### 1. Florence-2 OOM when running alongside OWLv2
- **Cause:** GPU only had ~2.5 GiB free while both owlv2 jobs occupied ~10 GiB
- **Fix:** Created `/tmp/run_florence2.sh` watcher script — polls GPU every 2 min, launches florence2_pos then florence2_neg sequentially after OWLv2 finishes

### 2. Three transformers 5.8.1 compatibility patches to HF cache
Florence-2 cached code was written for transformers 4.x, breaking with 5.8.1:

| Error | File | Fix |
|-------|------|-----|
| `AttributeError: 'Florence2LanguageConfig' has no attribute 'forced_bos_token_id'` | `configuration_florence2.py:265` | `getattr(self, "forced_bos_token_id", None)` |
| `AttributeError: RobertaTokenizer has no attribute additional_special_tokens` | `processing_florence2.py:89` | `getattr(tokenizer, "additional_special_tokens", [])` |
| `AttributeError: 'Florence2ForConditionalGeneration' has no attribute '_supports_sdpa'` | `modeling_florence2.py:2531` | Added `_supports_sdpa = False` and `_supports_flash_attn_2 = False` as class attributes |

## Key Facts
- Server env: `cs338` at `/home/counting/.conda/envs/cs338`, transformers 5.8.1
- Florence-2 HF cache: `/home/counting/.cache/huggingface/modules/transformers_modules/microsoft/Florence_hyphen_2_hyphen_large/21a599d414c4d928c9032694c424fb94458e3594/`
- OWLv2 took ~2h total; Florence-2 took ~4.5h each (sequentially)
- All 4 output JSONs contain 1286 entries (full val split)
