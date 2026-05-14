# Roadmap: Zero-Shot Object Counting — Experiments & Paper Revision

**Milestone:** M1 — Detector Ablation + Paper Introduction/Related Work
**Created:** 2026-05-12
**Core Value:** Empirically prove Rich Prompts generalize across open-world detectors; deliver polished Intro + Related Work sections.

## Progress

- [ ] **Phase 1: Server Setup & Baseline Reproduction**
- [ ] **Phase 2: Additional Detector Selection & Integration**
- [ ] **Phase 3: Exemplar Generation for Additional Detectors**
- [ ] **Phase 4: Val-Split Evaluation & Results Table**
- [ ] **Phase 5: Paper Rewrite — Introduction & Related Work**
- [ ] **Phase 6: Final Review & Submission Prep**

---

### Phase 1: Server Setup & Baseline Reproduction

**Goal:** Establish a reproducible Conda environment on `counting@192.168.6.200` (data at `/mnt/mmlab2024nas/counting`) and confirm the 4 existing pipeline configs reproduce their known val-split numbers.

**Depends on:** Nothing

**Requirements:** SRV-01, SRV-02, SRV-03, SRV-04

**Plans:** 0/2 plans executed

---

### Phase 2: Additional Detector Selection & Integration

**Goal:** Identify and integrate 1–2 additional open-world text-grounding detectors into the EEM pipeline.

**Depends on:** Phase 1

**Requirements:** DET-01, DET-02

**Plans:** TBD

---

### Phase 3: Exemplar Generation for Additional Detectors

**Goal:** Generate positive and negative exemplar JSONs for the full val split using the new detector(s), both with and without Rich Prompts.

**Depends on:** Phase 2

**Requirements:** DET-03, DET-04

**Plans:** TBD

---

### Phase 4: Val-Split Evaluation & Results Table

**Goal:** Run NSM inference for all detector × prompt combinations and compile the cross-detector ablation results table.

**Depends on:** Phase 3

**Requirements:** DET-05, DET-06, DET-07

**Plans:** TBD

---

### Phase 5: Paper Rewrite — Introduction & Related Work

**Goal:** Rewrite Introduction and Related Work from scratch with flowing, connected academic prose. Keep Abstract unchanged.

**Depends on:** Phase 4

**Requirements:** INT-01, INT-02, INT-03, INT-04, RW-01, RW-02, RW-03, RW-04

**Plans:** TBD

---

### Phase 6: Final Review & Submission Prep

**Goal:** Final proofread, cross-check all numbers, and prepare submission-ready PDF.

**Depends on:** Phase 5

**Requirements:** INT-01, RW-01

**Plans:** TBD

---
*Roadmap created: 2026-05-12*
