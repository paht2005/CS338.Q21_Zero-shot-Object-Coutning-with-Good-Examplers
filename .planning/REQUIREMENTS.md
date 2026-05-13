# Requirements: Zero-Shot Object Counting — Experiments & Paper Revision

**Defined:** 2026-05-12
**Core Value:** Empirically prove Rich Prompts reduce MAE across all open-world detectors, and deliver a well-written paper ready for MAPR 2026 submission.

## v1 Requirements

### Server Setup

- [ ] **SRV-01**: Server `phatcnguyen@sandbox.netbird.cloud` is accessible and environment is reproducible (Python venv)
- [ ] **SRV-02**: FSC-147 dataset is available under `code/source-code/data/FSC147/` (local, no NAS)
- [ ] **SRV-03**: Pre-trained NSM checkpoint is available on the server
- [ ] **SRV-04**: Existing 4 pipeline configs reproduce known val-split MAE/RMSE figures

### Detector Ablation

- [ ] **DET-01**: Identify and justify 1–2 additional open-world detectors compatible with the EEM (text prompt → bounding boxes)
- [ ] **DET-02**: Implement EEM adapter for each additional detector (positive + negative stream)
- [ ] **DET-03**: Generate positive/negative exemplar JSONs for val split with each additional detector (raw prompt)
- [ ] **DET-04**: Generate positive/negative exemplar JSONs for val split with each additional detector + Rich Prompts
- [ ] **DET-05**: Run NSM inference on val split for all detector × prompt configurations
- [ ] **DET-06**: Compile comparison table (MAE, RMSE per detector, ± Rich Prompt)
- [ ] **DET-07**: Demonstrate Rich Prompts reduce MAE for every tested detector (≥ val split evidence)

### Paper — Introduction

- [ ] **INT-01**: Introduction flows as connected prose: problem motivation → current gaps → our approach → contributions summary
- [ ] **INT-02**: No isolated bullet-list paragraphs; each paragraph leads into the next with explicit transitions
- [ ] **INT-03**: Contribution bullets (if kept) are embedded naturally within flowing text, not dropped as a standalone list
- [ ] **INT-04**: Matches IEEE IEEEtran style and column width

### Paper — Related Work

- [ ] **RW-01**: Related Work narrates the evolution of counting approaches rather than listing them independently
- [ ] **RW-02**: Each subsection ends by pointing to a limitation that the next subsection (or our method) addresses
- [ ] **RW-03**: Our contribution is introduced organically from the gaps identified in prior work
- [ ] **RW-04**: Writing is in connected academic prose — no isolated idea→explanation→problem blocks

## Out of Scope

| Feature | Reason |
|---------|--------|
| Abstract rewrite | Author instruction: keep as-is |
| NSM fine-tuning on new detectors | Out of compute budget for this milestone |
| Test-split evaluation for new detectors | Val split sufficient; test split reserved for final results |
| Rewriting Methodology / Results / Discussion / Conclusion | Not requested; addressed in a future pass |
| New detectors that don't support text-prompt grounding | Not compatible with EEM without major refactoring |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| SRV-01 – SRV-04 | Phase 1 | Pending |
| DET-01 – DET-02 | Phase 2 | Pending |
| DET-03 – DET-04 | Phase 3 | Pending |
| DET-05 – DET-07 | Phase 4 | Pending |
| INT-01 – INT-04 | Phase 5 | Pending |
| RW-01 – RW-04 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0

---
*Requirements defined: 2026-05-12*
*Last updated: 2026-05-12 after initialization*
