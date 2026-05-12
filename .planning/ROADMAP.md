# Roadmap: Zero-Shot Object Counting — Experiments & Paper Revision

**Milestone:** M1 — Detector Ablation + Paper Introduction/Related Work
**Created:** 2026-05-12
**Core Value:** Empirically prove Rich Prompts generalize across open-world detectors; deliver polished Intro + Related Work sections.

---

## Phase 1 — Server Setup & Baseline Reproduction

**Goal:** Establish a reproducible environment on `counting@192.168.6.200` and confirm the 4 existing configs reproduce their known val-split numbers.

**Depends on:** Nothing

### Plans

1. **Connect to server and inspect environment**
   - SSH in, verify GPU (nvidia-smi), disk space on `/mnt/mmlab2024nas/counting`, available Conda/Docker
   - Check if FSC-147 is already present; if not, download to `/mnt/mmlab2024nas/counting/fsc147/`
   - Check if NSM checkpoint is present; if not, download from VA-Count release

2. **Reproduce Conda environment**
   - Create Conda env from `requirements.txt` (or build Docker image if Conda fails)
   - Verify all imports: `torch`, `clip`, `ultralytics`, `groundingdino`, `google.generativeai`
   - Confirm `yolov8x-worldv2.pt` checkpoint is present

3. **Run val-split baseline evaluation (4 configs)**
   - Run `FSC_test.py` on val split for all 4 pre-generated exemplar sets
   - Record MAE and RMSE; compare to expected test-split values (within ~1 MAE acceptable)
   - Commit results to `experiments/server-val-baseline/`

**UAT:**
- [ ] `nvidia-smi` shows GPU on server
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` returns `True`
- [ ] Val-split MAE for baseline (GDino, raw) is within 1 point of 17.99
- [ ] All 4 configs produce MAE/RMSE without errors

---

## Phase 2 — Additional Detector Selection & Integration

**Goal:** Identify and integrate 1–2 additional open-world text-grounding detectors into the EEM pipeline.

**Depends on:** Phase 1

### Plans

1. **Evaluate candidate detectors**
   - Candidates: OWLv2 (Google, HuggingFace), GLIP (Microsoft), Detic, MDETR
   - Criteria: (a) text prompt → bounding box output, (b) installable in existing env, (c) model size fits VRAM, (d) publicly available checkpoint
   - Write selection rationale to `experiments/detector-candidates/SELECTION.md`

2. **Implement EEM adapter for chosen detector(s)**
   - Create `code/source-code/owlv2_pos.py` and `owlv2_neg.py` (or equivalent) following the same interface as `grounding_pos.py` / `yolo_pos_withPrompt.py`
   - Adapt: (a) model loading, (b) inference call, (c) box format normalization, (d) CLIP re-ranking hook
   - Unit test: run on 5 sample FSC-147 images, confirm box detections are returned

3. **Validate integration end-to-end**
   - Run the full pipeline (detector → EEM → NSM) on 20 val images per config
   - Confirm no shape/dtype errors in exemplar patches passed to NSM

**UAT:**
- [ ] `SELECTION.md` documents chosen detector(s) with rationale
- [ ] New `*_pos.py` / `*_neg.py` scripts produce valid bounding boxes on sample images
- [ ] End-to-end pipeline runs on 20 val images with no runtime errors

---

## Phase 3 — Exemplar Generation for Additional Detectors

**Goal:** Generate positive and negative exemplar JSONs for the full val split using the new detector(s), both with and without Rich Prompts.

**Depends on:** Phase 2

### Plans

1. **Generate exemplars — new detector, raw prompt**
   - Run positive and negative exemplar generation for all val-split images
   - Output to `/mnt/mmlab2024nas/counting/exemplars/<detector>/raw/`
   - Log wall-clock time for extraction

2. **Generate exemplars — new detector, Rich Prompt**
   - Use pre-generated Gemini prompts (positive and negative descriptions) from existing FSC-147 runs if available; generate only missing classes
   - Run positive and negative extraction with Rich Prompts
   - Output to `/mnt/mmlab2024nas/counting/exemplars/<detector>/rich/`
   - Log wall-clock time

3. **Quality check**
   - Spot-check 10 random images per config: open exemplar JSON, verify crops look correct
   - Report number of images with zero detections per config

**UAT:**
- [ ] Both `raw/` and `rich/` exemplar JSONs exist for all val-split images
- [ ] Zero-detection rate is documented and is not unreasonably high (< 15%)
- [ ] Wall-clock extraction time is recorded

---

## Phase 4 — Val-Split Evaluation & Results Table

**Goal:** Run NSM inference for all detector × prompt combinations and compile the cross-detector ablation results table.

**Depends on:** Phase 3

### Plans

1. **Run NSM inference — all configs**
   - For each (detector, prompt-mode) pair, run `FSC_test.py` on val split
   - Collect MAE and RMSE; save JSON results to `experiments/cross-detector/`
   - Configs: {GDino-raw, GDino-rich, YOLO-raw, YOLO-rich, <new>-raw, <new>-rich}

2. **Compile results table**
   - Create `experiments/cross-detector/RESULTS.md` with Table: detector × prompt → MAE, RMSE, ΔMAE (rich vs raw), extraction time
   - Verify Rich Prompts reduce MAE for every detector (≥ on val split)

3. **Write analysis paragraph**
   - Summarize the pattern (Rich Prompts consistently reduce MAE; magnitude varies by detector semantic capacity)
   - Draft text for a new subsection in the paper (to be inserted into Results section)

**UAT:**
- [ ] Results JSON exists for all (detector, prompt) configs on val split
- [ ] RESULTS.md table is complete and shows Rich Prompt reduces MAE for every detector
- [ ] Analysis paragraph drafted (≥ 100 words, references ΔMAE numbers)

---

## Phase 5 — Paper Rewrite: Introduction & Related Work

**Goal:** Rewrite Introduction and Related Work from scratch with flowing, connected academic prose. Keep Abstract unchanged.

**Depends on:** Phase 4 (for updated results context)

### Plans

1. **Draft Introduction — flowing narrative**
   - Structure: (1) motivate the task with real-world importance, (2) describe the challenge of zero-shot counting, (3) identify the specific gap our work addresses (prompt quality + speed), (4) state our approach and contributions as connected prose (not a dropped bullet list), (5) one-sentence chapter guide
   - Read 2–3 reference papers (VA-Count, CLIP-Count, ZSC) to absorb academic tone
   - Target: ~500–600 words; no isolated bullet blocks

2. **Draft Related Work — narrative evolution**
   - Structure: a single narrative arc from density regression → few-shot exemplar counting → zero-shot open-vocabulary counting → our work
   - Each subsection ends with an explicit limitation that motivates the next
   - Subsection headings changed from "Previous Approach N" to descriptive titles
   - Target: ~600–700 words; prose-first, math only where necessary

3. **Integrate cross-detector results into paper**
   - Add a new paragraph/row in the Results section for cross-detector ablation (from Phase 4 analysis)
   - Update abstract claim footnote or cite the new table appropriately

4. **LaTeX compile check**
   - Compile `main.tex` with `latexmk` or `xelatex`; confirm no errors and page count is within IEEE 6-page limit
   - Check figures and references still resolve correctly

**UAT:**
- [ ] Introduction reads as a single connected narrative (no isolated bullet lists)
- [ ] Related Work subsections each connect explicitly to the next via transitional sentences
- [ ] LaTeX compiles without errors
- [ ] Paper is within 6 IEEE pages (or within target page budget)
- [ ] Abstract is unchanged

---

## Phase 6 — Final Review & Submission Prep

**Goal:** Final proofread, cross-check all numbers, and prepare submission-ready PDF.

**Depends on:** Phase 5

### Plans

1. **Proofread and consistency check**
   - Verify all MAE/RMSE numbers in body text match tables
   - Check all `\cite{}` keys resolve; no missing references
   - Proofread Introduction and Related Work for grammar and style

2. **Submission prep**
   - Generate final PDF (`latexmk -pdf main.tex`)
   - Check PDF metadata, author info, IEEEtran format compliance
   - Archive final PDF to `docs/report/` and tag the git commit

**UAT:**
- [ ] All numbers consistent across abstract, tables, body text
- [ ] Final PDF generated without warnings
- [ ] Git tag `v1.0-mapr2026` created on submission commit

---

## Summary

| Phase | Focus | Key Output |
|-------|-------|------------|
| 1 | Server setup & baseline reproduction | Working env, val-split baseline MAE |
| 2 | Additional detector integration | EEM adapters for new detector(s) |
| 3 | Exemplar generation | Val-split exemplar JSONs (raw + rich) |
| 4 | Cross-detector evaluation | Results table proving Rich Prompts generalize |
| 5 | Paper rewrite (Intro + Related Work) | Polished, flowing narrative sections |
| 6 | Final review & submission | Submission-ready PDF |

---
*Roadmap created: 2026-05-12*
*Next action: `/gsd-plan-phase 1`*
