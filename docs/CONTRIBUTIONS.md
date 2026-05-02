# Contributions — CS338.Q21

This file lists the per-member contributions for the CS338.Q21 (Pattern
Recognition, UIT) project *“Zero-shot Object Counting with Good Exemplars”*.
The same breakdown also appears as Table `tab:phan_cong` in
[`docs/report/Report.pdf`](report/Report.pdf) (TEAM MEMBER INFORMATION).

## Team

| No. | Student ID | Full Name        | Role        | GitHub                                               | Email                       |
|----:|:----------:|------------------|-------------|------------------------------------------------------|-----------------------------|
| 1   | 23521143   | Nguyen Cong Phat | Leader      | [paht2005](https://github.com/paht2005)              | 23521143@gm.uit.edu.vn      |
| 2   | 23520158   | Mai Thai Binh    | Member      | [maibinhkznk209](https://github.com/maibinhkznk209/) | 23520158@gm.uit.edu.vn      |
| 3   | 23520213   | Vu Viet Cuong    | Member      | [Kun05-AI](https://github.com/Kun05-AI)              | 23520213@gm.uit.edu.vn      |

## Contribution split

| Member             | Share | Primary deliverables                                                                                                                                                                                                                                                                                                                          |
|--------------------|------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Nguyen Cong Phat** (Leader) | 35%   | Repository structure (`docs/`, `configs/`, `scripts/`); training pipeline integration (`FSC_train.py`, `FSC_test.py`); secrets handling for the Gemini API key (`GEMINI_API_KEY` via `.env`); wrote Chapter 1 (Overview) and Chapter 2 (Methods) of `Report.pdf`; managed Git history and reviews. |
| **Mai Thai Binh**             | 35%   | Implemented the **Rich Prompt module** (Gemini 2.0 Flash) and the `grounding_pos.py` / `grounding_neg.py` pipelines; ran the MAE / RMSE evaluation sweeps and built the speed-vs-accuracy chart; reconciled the values against the project's `wandb` runs under `experiments/exp{2,3,4,5}/wandb/`; wrote Chapter 3 §3.4 (Rich Prompt results) and the failure-analysis sub-section. |
| **Vu Viet Cuong**             | 30%   | Implemented the **YOLO-World extension** (`yolo_pos_withPrompt.py`, `yolo_pos_withoutPrompt.py`, `yolo_neg.py`); end-to-end testing and documentation of the Streamlit demo (`demo_app_advanced.py` + supporting modules `demo_inference.py`, `demo_pipeline_advanced.py`, `demo_visualization.py`); wrote Chapter 3 §3.5 (YOLO-World), Chapter 4 (Demo) and Chapter 5 (Conclusion) of `Report.pdf`; produced the presentation slides. |
| **Total**                     | **100%** |                                                                                                                                                                                                                                                                                                                                            |

The split reflects the realised work between Nov 2025 and Apr 2026 and was
agreed unanimously by all members.

## Scope of the project

This project re-implements the ECCV 2024 **VA-Count** paper from scratch and
extends it with two independent additions from the literature: **Rich Prompts**
(Zhu et al., 2025) for higher-quality exemplar generation, and **YOLO-World**
(Cheng et al., 2024) as a fast drop-in replacement for GroundingDINO in the
exemplar-extraction stage. The team's deliverables span:

- **Code**: the full training, evaluation, exemplar-generation and demo
  pipelines under `code/source-code/`.
- **Experiments**: the wandb runs archived under
  `experiments/exp{2,3,4,5}/wandb/` (cross-checked against the surviving
  `wandb-summary.json` files).
- **Documentation**: a 26-page bilingual LaTeX report
  ([`docs/report/Report.pdf`](report/Report.pdf)), this contributions file,
  [`RESULTS.md`](RESULTS.md), the
  [`docs/report/figures/README.md`](report/figures/README.md) figure
  provenance file, the [`docs/report/README.md`](report/README.md) build
  guide, and the project root README.

## Acknowledgements

- Upstream open-source projects we built upon:
  [VA-Count](https://arxiv.org/abs/2407.04948),
  [Rich Prompts for Counting](https://arxiv.org/abs/2505.15398),
  [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO),
  [YOLO-World](https://github.com/AILab-CVC/YOLO-World),
  [CounTR](https://github.com/Verg-Avesta/CounTR), and
  [MAE](https://github.com/facebookresearch/mae).
- **Course supervisor**: TS. Duong Viet Hang (Faculty of Computer Science, UIT) —
  we sincerely thank the instructor for her guidance and feedback throughout CS338.Q21.

## How contributions land in Git

The repository is small enough that everyone pushes directly to `main` after
review on the group chat; squash-merge PRs are used only for larger
restructures (Wave A and the LaTeX re-write). The Git author / commit history
is the canonical record of who touched which file — `git log --author=...`
will reproduce the per-member breakdown if needed.
