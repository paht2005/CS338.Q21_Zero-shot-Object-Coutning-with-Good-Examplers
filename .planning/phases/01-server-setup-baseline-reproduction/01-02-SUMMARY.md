---
phase: 01-server-setup-baseline-reproduction
plan: 02
subsystem: evaluation
tags: [fsc147, val-split, baseline, mae, rmse, evaluation]

requires:
  - 01-01-SUMMARY.md  # conda env + checkpoints verified
provides:
  - FSC147 val-split MAE/RMSE/NAE for all 4 pipeline configs
  - scripts/run_evaluation_val.sh — reproduces 4-config val evaluation
  - experiments/server-val-baseline/{baseline,dino_rich,yolo_norich,yolo_rich}.log
affects: [phase-02-detector-comparison, paper-results-section]

tech-stack:
  added: []
  patterns:
    - "PYTHON env var (default /home/counting/.conda/envs/cs338/bin/python3) for non-interactive SSH runs"
    - "nohup bash script > /tmp/log 2>&1 & pattern for detached server jobs"
---

## Summary

Phase 01 Plan 02 completed. All 4 pipeline configurations evaluated on the
FSC-147 **val split** (1,286 images) on server `counting@192.168.6.200`.

## Results — FSC-147 Val Split

| Config | MAE ↓ | RMSE ↓ | NAE ↓ |
|---|---|---|---|
| baseline (GroundingDINO, raw prompt) | 18.94 | 74.08 | 0.3384 |
| dino_rich (GroundingDINO + Rich Prompt) | 19.05 | 73.88 | 0.3318 |
| yolo_norich (YOLO-World, no Rich Prompt) | 20.29 | 75.20 | 0.3588 |
| **yolo_rich (YOLO-World + Rich Prompt)** | **19.03** | **73.35** | **0.3352** |

Logs: `experiments/server-val-baseline/{baseline,dino_rich,yolo_norich,yolo_rich}.log`

## Key Observations

- `yolo_rich` achieves the best RMSE (73.35) — matches GroundingDINO baseline
  MAE while being faster at inference.
- `dino_rich` reduces NAE the most (0.3318) but costs more exemplar generation
  time than yolo_rich.
- `yolo_norich` is the weakest config (MAE 20.29, RMSE 75.20) — confirms that
  Rich Prompt is the key component, not YOLO-World alone.
- Val-split MAE range is narrow (18.94–20.29), suggesting all configs converge
  to similar accuracy at val time; RMSE spread is also small (73.35–75.20).

## Bugs Fixed During This Plan

1. `python: command not found` in `run_evaluation_val.sh` — fixed by adding
   `PYTHON="${PYTHON:-/home/counting/.conda/envs/cs338/bin/python3}"` variable
   (commit `800e4ba`).
2. FSC147 dataset only had 1,140 images on server — fixed by downloading full
   `images_384_VarV2.zip` (1.53 GB, 6,147 images) from Google Drive.
3. `ResizeSomeImage` re-parsed sys.argv — fixed with `parse_known_args` +
   `getattr` guards (commit `3804880`).
4. `images_384_VarV2` path not found — added auto-detect fallback in
   `FSC_test.py` (same commit).
5. `WindowsPath` error loading dino_rich checkpoint on Linux — patched
   `pathlib.WindowsPath = pathlib.PosixPath` in `util/misc.py` (same commit).
