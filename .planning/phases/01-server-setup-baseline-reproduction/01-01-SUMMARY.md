---
phase: 01-server-setup-baseline-reproduction
plan: 01
subsystem: infra
tags: [conda, cuda, fsc147, nas, groundingdino, clip, ultralytics]

requires: []
provides:
  - Conda env cs338 on counting@192.168.6.200 with all dependencies (torch 2.5.1+cu121, clip, groundingdino, timm, ultralytics, google.generativeai)
  - scripts/server_setup.sh — reproducible 8-step Conda setup for mmlab server
  - scripts/verify_env.sh — 5-category smoke test (imports, GPU, NAS data, checkpoints, .env)
  - Verified NAS data: FSC147 images/annotations at /mnt/mmlab2024nas/counting/FSC147/
  - Verified 4 checkpoints: checkpoint_FSC.pth, checkpoint__finetuning_dino_prompt.pth, checkpoint__finetuning_yolo.pth, checkpoint__finetuning_yolo_noprompt.pth
affects: [baseline-reproduction, evaluation, detector-integration]

tech-stack:
  added: []
  patterns:
    - "DATA_BASE env var (default /mnt/mmlab2024nas/counting) for NAS path overrides"
    - "Conda env cs338 on counting@192.168.6.200 (not venv)"

key-files:
  created: []
  modified:
    - scripts/server_setup.sh
    - scripts/verify_env.sh

key-decisions:
  - "Conda (not venv) for env management on mmlab server"
  - "NAS path /mnt/mmlab2024nas/counting as DATA_BASE default"
  - "All 4 checkpoints confirmed present on NAS before proceeding to eval"

patterns-established:
  - "DATA_BASE pattern: all scripts default to /mnt/mmlab2024nas/counting, overridable via env var"
  - "verify_env.sh pattern: 5-category check with per-check PASS/FAIL + summary count"

requirements-completed:
  - SRV-01
  - SRV-02
  - SRV-03

duration: ~30min (script updates + server run)
completed: 2026-05-14
---

# Phase 01 / Plan 01: Server Setup Summary

**Conda env `cs338` confirmed on counting@192.168.6.200 — RTX 4090, NAS FSC-147 data and all 4 checkpoints verified (13/13 checks passed).**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-05-14
- **Completed:** 2026-05-14
- **Tasks:** 3/3 (2 automated + 1 human-action checkpoint)
- **Files modified:** 2

## Accomplishments

- Updated `scripts/server_setup.sh` for Conda + NAS paths on counting@192.168.6.200 (8-step setup)
- Updated `scripts/verify_env.sh` with 5-category smoke test (imports, GPU, NAS, checkpoints, .env)
- User ran `verify_env.sh` on server → **13/13 checks passed**: torch 2.5.1+cu121, CUDA (RTX 4090), all FSC-147 NAS paths, all 4 `.pth` checkpoints, `.env` with GEMINI_API_KEY

## Task Commits

1. **Task 1: Verify server_setup.sh** — `f2b5464` (docs(01): update server to counting@192.168.6.200, NAS data path, Conda env)
2. **Task 2: Verify verify_env.sh** — `4150e74` (fix(01): hardcode conda base to /opt/miniconda3 on mmlab server)
3. **Task 3: Human checkpoint** — server confirmed manually (no commit needed)

## Files Created/Modified

- `scripts/server_setup.sh` — 8-step Conda env bootstrap for counting@192.168.6.200
- `scripts/verify_env.sh` — 5-category smoke test with PASS/FAIL output and NAS data checks

## Server State (verified)

| Check | Result |
|-------|--------|
| torch 2.5.1+cu121, CUDA | ✅ PASS |
| clip, ultralytics, groundingdino, timm, google.generativeai | ✅ PASS (all) |
| CUDA device | ✅ NVIDIA GeForce RTX 4090 |
| /mnt/mmlab2024nas/counting/FSC147/images_384_VarV2 | ✅ PASS |
| /mnt/mmlab2024nas/counting/FSC147/gt_density_map_adaptive_384_VarV2 | ✅ PASS |
| Train_Test_Val_FSC_147.json | ✅ PASS |
| annotation_FSC147_384.json | ✅ PASS |
| checkpoint_FSC.pth | ✅ PASS |
| checkpoint__finetuning_dino_prompt.pth | ✅ PASS |
| checkpoint__finetuning_yolo.pth | ✅ PASS |
| checkpoint__finetuning_yolo_noprompt.pth | ✅ PASS |
| .env with GEMINI_API_KEY | ✅ PASS |
| **Total** | **13/13** |

## Self-Check: PASSED

- [x] scripts/server_setup.sh passes `bash -n`, references `mmlab2024nas` + `CONDA_ENV`, no old venv/sandbox refs
- [x] scripts/verify_env.sh passes `bash -n`, 5 check categories with PASS/FAIL, DATA_BASE defaults to /mnt/mmlab2024nas/counting
- [x] User confirmed 13/13 checks passed on server
