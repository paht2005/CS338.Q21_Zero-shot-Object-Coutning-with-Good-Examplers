# Zero-Shot Object Counting — Experiments & Paper Revision

## What This Is

A research project extending VA-Count with Rich Prompts and YOLO-World for zero-shot object counting on FSC-147. The current milestone covers two tracks: (1) broadening the detector ablation to additional open-world detectors on the **validation split** to empirically back the abstract claim that "Rich Prompts consistently reduce MAE regardless of the underlying detector", and (2) rewriting the Introduction and Related Work sections of the MAPR 2026 paper to replace list-style writing with cohesive, connected prose.

## Core Value

Validate the scalability claim of Rich Prompts across detectors **and** have a paper ready for submission with well-written, flowing narrative sections.

## Requirements

### Validated

- ✓ Rich Prompt pipeline (Gemini + CLIP re-ranking) implemented — existing
- ✓ YOLO-World drop-in detector implemented — existing
- ✓ GroundingDINO baseline implemented — existing
- ✓ NSM pretrained checkpoint available — existing
- ✓ FSC-147 dataset available at `code/source-code/data/FSC147/` on sandbox — existing (to be uploaded)

### Active

- [ ] Run val-split evaluation of all 4 baseline configs on server (counting@192.168.6.200)
- [ ] Identify 1–2 additional open-world detectors compatible with the EEM pipeline
- [ ] Generate exemplars and run val-split evaluation for additional detectors (with and without Rich Prompts)
- [ ] Compile results table showing Rich Prompts reduce MAE across all detectors
- [ ] Rewrite Introduction section — problem → challenges → approach → summary (connected prose)
- [ ] Rewrite Related Work section — narrative leading from prior limitations to our contribution

### Out of Scope

- Rewriting Abstract — keep as-is per instructions
- Training / fine-tuning NSM on new detectors — too expensive, not in scope
- Test-split evaluation for new detectors — val split sufficient to prove the claim
- Rewriting Methodology, Results, Discussion, Conclusion — addressed separately if needed

## Context

- **Paper target**: MAPR 2026 (IEEE conference format, IEEEtran)
- **Server**: `ssh counting@192.168.6.200` (pass: 1), has Docker + Conda, GPU TBD
- **Data location on server**: `/mnt/mmlab2024nas/counting` (not `/home/`)
- **GPU**: RTX 4060 on the lab workstation (existing experiments), server GPU TBD
- **Existing results** (test split):
  - VA-Count baseline (GDino, raw): MAE 17.99, RMSE 129.39
  - VA-Count + Rich Prompt (GDino): MAE 17.80, RMSE 129.69
  - VA-Count + YOLO-World: MAE 19.03, RMSE 131.55
  - VA-Count + YOLO-World + Rich Prompt: MAE 17.91, RMSE 130.98
- **Candidate additional detectors**: OWL-ViT (Google), GLIP, Detic, OWLv2, MDETR — need to evaluate feasibility
- **Current writing issues**: sections read as bulleted lists rather than flowing academic prose; each paragraph isolated rather than connected

## Constraints

- **Hardware**: Must use `/mnt/mmlab2024nas/counting` for data on server, not `/home/`
- **Compute**: Server environment: Conda (primary) or Docker — both available on server
- **Paper style**: IEEE conference (IEEEtran), keep abstract unchanged
- **Detector compatibility**: New detectors must support text-prompt → bounding-box output to plug into existing EEM

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use val split (not test) for additional detectors | Test split already used for main results; val split avoids overfitting the claim | — Pending |
| Keep Abstract unchanged | Mentor/author instruction | ✓ Good |
| Rewrite from scratch (not edit) | Current prose too list-like to fix incrementally | — Pending |
| Run on remote server | Local GPU insufficient for parallel multi-detector runs | counting@192.168.6.200 confirmed |

---
*Last updated: 2026-05-14 — server updated to counting@192.168.6.200, data at /mnt/mmlab2024nas/counting, Conda environment*
