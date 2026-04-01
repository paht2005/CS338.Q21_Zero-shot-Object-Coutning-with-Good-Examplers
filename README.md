<p align="center">
  <a href="https://www.uit.edu.vn/" title="University of Information Technology">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="University of Information Technology (UIT)" width="400">
  </a>
</p>

<h1 align="center"><b>Zero-shot Object Counting with Good Exemplars</b></h1>
<h3 align="center">Enhanced with Rich Prompts and YOLO-World</h3>
<h4 align="center">CS338.Q21 – Pattern Recognition</h4>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/GroundingDINO-000000?style=for-the-badge" />
  <img src="https://img.shields.io/badge/YOLO--World-00A67E?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Gemini_Prompts-4285F4?style=for-the-badge" />
</p>

---

# **Project Overview**

> This repository contains the implementation of a **Zero-shot Object Counting system**  
> based on **VA-Count (Zero-shot Object Counting with Good Exemplars)**, extended with:
>
> - **Rich Prompts**: Large Language Model–generated prompts (Gemini) + CLIP-style re-ranking  
>   to obtain higher-quality positive and negative exemplars.
> - **YOLO-World**: A fast open-vocabulary detector used to replace GroundingDINO in the
>   exemplar extraction stage, dramatically reducing latency while preserving accuracy.
>
> The system targets **generalized object counting in unseen classes** using **FSC147**,  
> and is evaluated with standard metrics (**MAE, RMSE**) and detailed latency measurements.

The full written report is stored at the root as: **`Report.pdf`**.  
Full project artifacts (including large checkpoints and raw experiment logs) are hosted on OneDrive:  
**[Full project on OneDrive](YOUR_ONEDRIVE_LINK_HERE)**.

---

<p align="center">
  <img src="images/thumbnail.png" alt="Project thumbnail" width="400">
</p>

---

## Team Information

| No. | Student ID | Full Name        | Role   | Github                                                   | Email                   |
|----:|:----------:|------------------|--------|----------------------------------------------------------|-------------------------|
| 1   | 23521143   | Nguyen Cong Phat | Leader | [paht2005](https://github.com/paht2005)                  | 23521143@gm.uit.edu.vn |
| 2   | 23520158   | Mai Thai Binh    | Member | [maibinhkznk209](https://github.com/maibinhkznk209/)     | 23520158@gm.uit.edu.vn |
| 3   | 23520213   | Vu Viet Cuong    | Member | [Kun05-AI](https://github.com/Kun05-AI)                  | 23520213@gm.uit.edu.vn |

---

## **Table of Contents**

- [Repository Structure](#repository-structure)
- [Problem Statement](#problem-statement)
- [Method Overview](#method-overview)
  - [Baseline VA-Count (EEM + NSM)](#baseline-va-count-eem--nsm)
  - [Rich Prompt Extension](#rich-prompt-extension)
  - [YOLO-World Extension](#yolo-world-extension)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Experiments & Metrics](#experiments--metrics)
- [Installation](#installation)
- [Usage](#usage)
- [Demo Application](#demo-application)
- [Experimental Results (High-level)](#experimental-results-high-level)
- [Limitations & Future Work](#limitations--future-work)
- [License](#license)

---

## **Repository Structure**

```text
ZeroShot-Object-Counting-with-Good-Examplers/
├── README.md                # Main project documentation (this file)
│
├── docs/                    # Course reports and documents (PDF, slides, LaTeX, etc.)
│
├── images/                  # Images for GitHub (thumbnails, screenshots, GIF demos)
│   └── Thumbnail.png        # Main project thumbnail
│
├── code/                    # Main implementation (training, inference, demos)
│   ├── README.md
│   └── source-code/
│       ├── README.md        # Detailed instructions (env, data, checkpoints)
│       ├── data/            # FSC147 structure (images, density maps, annotations)
│       ├── GroundingDINO/   # GroundingDINO code + weights
│       ├── util/            # Dataset loader, misc utilities
│       ├── models_*.py      # MAE / CrossViT model definitions
│       ├── FSC_pretrain.py  # Pretraining script
│       ├── FSC_train.py     # Training / fine-tuning script
│       ├── FSC_test.py      # Evaluation script (MAE, RMSE, etc.)
│       ├── prompt_enhancer.py
│       ├── grounding_*.py   # GroundingDINO-based exemplar extraction
│       ├── yolo_*.py        # YOLO-World–based exemplar extraction
│       ├── demo_*.py        # Demo & visualization scripts
│       └── requirements.txt # Python dependencies
│
└── experiments/             # Archived experiment runs and logs
    ├── README.md
    ├── exp2/
    ├── exp3/
    ├── exp4/
    └── exp5/
```

- **`code/`** – Everything needed to **run the models**:
  - Baseline VA-Count (GroundingDINO + MAE + NSM)
  - Rich Prompt pipeline
  - YOLO-World based exemplar extraction
  - Demo applications and visualization scripts  
  See `code/README.md` and `code/source-code/README.md` for details.

- **`experiments/`** – Data snapshots, GroundingDINO installs, YOLO-World weights,
  visualizations and `wandb` logs for different experimental setups that correspond to
  the ablations and results reported in `Report.pdf`.  
  See `experiments/README.md` and `experiments/exp*/README.md` for per-run notes.

- **`docs/`** – Course-related documents:
  - Final project report (PDF)
  - Presentation slides
  - LaTeX sources or any other classroom submission material.

- **`images/`** – Assets for the GitHub repository:
  - Thumbnails
  - Screenshots
  - Animated GIFs demonstrating live demos.

---

## **Problem Statement**

Counting objects in natural images is challenging due to:

- Wide variability in **object scale, density, and occlusion**.
- The need to generalize to **unseen classes** without additional annotations.
- The cost of **dense labeling** (dot annotations, bounding boxes) at scale.

The project addresses **Zero-shot Object Counting**:

> Given an RGB image and a **text prompt** describing the target class  
> (e.g. `"person"`, `"oranges"`), estimate:
> - A **density map** over the image.
> - The **total count** of objects that match the prompt.

Under the constraints:

- Only a **single target class** is counted per run.
- Prompts are **English**, concrete, countable nouns.
- Objects are visually observable in real-world scenes.

---

## **Method Overview**

The implementation follows VA-Count and extends it with two major components.

### Baseline VA-Count (EEM + NSM)

Baseline pipeline (Section 4.1 in the report):

- **Exemplar Enhancement Module (EEM)**:
  - Uses **GroundingDINO** to propose **positive** and **negative** exemplar boxes.
  - Filters boxes via:
    - **Deduplication** across positive/negative streams (IoU-based).
    - **Binary classifier** for **single-object filtering**.
  - Selects top exemplars for both positive and negative branches.

- **Noise Suppression Module (NSM)**:
  - MAE-based image encoder + interaction module + decoder.
  - Produces:
    - Positive density map `Dp`
    - Negative density map `Dn`
  - Trained with:
    - **Contrastive loss** `LC(Dp, Dg, Dn)`
    - **Density MSE loss** `LD(Dp, Dg)`
    - `L_total = LC + LD`

### Rich Prompt Extension

To reduce ambiguity and improve exemplar quality, the project introduces **Rich Prompts**:

- Uses **Gemini** to generate:
  - **Positive prompts**: visual definition of a *single* target instance
    (shape, color, material, etc.).
  - **Negative prompts**: background and distractor descriptions that must *not*
    describe the target class.
- Uses these prompts as **queries** for GroundingDINO.
- Adds **CLIP ViT-B/32 re-ranking** over the proposed boxes:
  - Patches from candidate boxes vs. class name text embeddings.
  - Selects top-k (e.g. 5) patches with highest CLIP similarity as final exemplars.

### YOLO-World Extension

GroundingDINO is accurate but **slow and heavy**.  
To make exemplar extraction real-time friendly, the project explores **YOLO-World**:

- Replaces GroundingDINO proposals with **YOLO-World detections**.
- Leverages YOLO-World’s **Prompt-then-Detect** design and **RepPAN** backbone.
- Achieves significantly higher FPS while preserving acceptable accuracy.
- Combined with Rich Prompts, the model reaches a good trade-off between:
  - **Accuracy** (MAE/RMSE on FSC147)
  - **Latency** (seconds per image in the demo)

---

## **Key Features**

- Zero-shot object counting on **unseen classes** via text prompts.
- Strong baseline **VA-Count** implementation (EEM + NSM).
- **Rich Prompt** generation and CLIP-based re-ranking for robust exemplars.
- **YOLO-World** integration for fast exemplar extraction (20× speedup region).
- Full experimental pipeline:
  - FSC147 dataset preparation
  - Training, pretraining, and testing scripts
  - Demo and visualization tools
  - Archived experiment logs (`wandb`, visualizations)

---

## **Dataset**

The project uses **FSC147**, a standard dataset for generalized object counting:

- ~6,135 images, **147 classes**.
- High intra-class variation in:
  - Object size
  - Density
  - Background complexity
- Provides:
  - Class labels
  - Dot annotations
  - Train/Val/Test splits

Official dataset (UIT-hosted Kaggle copy) used in this project:

- Kaggle dataset: `https://www.kaggle.com/datasets/xuncngng/fsc147-0`

Expected structure under `code/source-code/data/FSC147/` is documented in  
`code/source-code/README.md` (images, density maps, annotations, splits).

### Dataset in this repository

Because FSC147 is relatively large and subject to its own license, **this repository does not contain the full dataset**.  
Instead, only a **very small sample** is tracked to illustrate the expected directory layout:

- `code/source-code/data/FSC147/sample/`
  - Contains **2–5 example images** from FSC147.
  - Each sample image is paired with its corresponding **dot-annotation / metadata file**
    following the **official FSC147 folder and file naming conventions**.

To run training or full evaluation you must:

1. Download the full dataset from the Kaggle link above.
2. Unzip it and place the content under:

   - `code/source-code/data/FSC147/`

3. Make sure the folder hierarchy (images, annotations, density maps, splits, etc.) matches
   what is described in `code/source-code/README.md`.

The `.gitignore` is configured so that:

- The **full FSC147 dataset** under `code/source-code/data/FSC147/` is **ignored by git**.
- Only the tiny **`sample/`** subfolder (2–5 images + annotations) is included in version control
  as a reference example.

---

## **What is tracked in Git vs. kept locally**

To make the repository lightweight and friendly for GitHub (no files larger than 100 MB),
we separate **source code and small samples** from **heavy artifacts**.

- **Tracked in Git (safe to clone / view on GitHub)**:
  - All Python source code under `code/source-code/` (models, training, testing, demos, utilities).
  - Configuration files and scripts.
  - Project documentation (`README.md`, `docs/`, `images/`).
  - A tiny FSC147 sample under `code/source-code/data/FSC147/sample/`
    (2–5 images + their annotations) to illustrate the expected data layout.

- **Ignored by Git (large or environment-specific artifacts)**:
  - Full FSC147 dataset under `code/source-code/data/FSC147/` (only `sample/` is kept).
  - All checkpoints and large model weights (`*.pth`, `*.pt`, `*.ckpt`, `*.onnx`).
  - Training / evaluation outputs:
    - `code/source-code/output/`, `code/source-code/output_transfer/`,
      `code/source-code/data/out/`, `demo_outputs/`.
    - Experiment outputs under `experiments/**/data/out/` and `experiments/**/visualizations/`.
  - Visualization artifacts:
    - `code/source-code/visualizations/`, `code/source-code/visualizations_test/`.
  - Experiment tracking logs:
    - `wandb/`, `code/source-code/wandb/`, `experiments/**/wandb/`.
  - Heavy experiment-only weights:
    - `experiments/**/GroundingDINO/weights/`, `experiments/**/yolo*.pt`.
  - Very large visualization notebooks under `experiments/**/notebook/`.
  - Any temporary copies of FSC147 placed under experiments/**/data/FSC147/ are also ignored by git and are meant only for local experiments.

All of these ignore rules are defined in `.gitignore` so that **pushing to GitHub never uploads
large datasets or checkpoints**, and contributors only interact with the **code, docs, and small
sample data**.

---

## **Experiments & Metrics**

The project follows the report:

- **Metrics**:
  - **MAE** (Mean Absolute Error)
  - **RMSE** (Root Mean Squared Error)
- **Comparisons**:
  - Baseline VA-Count
  - VA-Count + Rich Prompts
  - VA-Count + YOLO-World
  - VA-Count + YOLO-World + Rich Prompts
- **Artifacts**:
  - Numerical results and plots: logged via `wandb` in `experiments/exp*/wandb/`.
  - Qualitative results and failure cases: stored in `experiments/exp*/visualizations/`.

---

## **Installation**

1. **Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/ZeroShot-Object-Counting.git
cd ZeroShot-Object-Counting
```

2. **Create and activate a virtual environment (recommended)**

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows
```

3. **Install Python dependencies**

```bash
cd code/source-code
pip install -r requirements.txt
```

4. **Install GroundingDINO (editable)**

```bash
cd GroundingDINO
pip install -e .
cd ..
```

5. **Prepare the FSC147 dataset**

Follow the **Dataset Preparation** section in `code/source-code/README.md`:

- Download FSC147 from the official repository.
- Place it under `data/FSC147/`.
- Run the split-generation script to create `train.txt`, `val.txt`, `test.txt`.

6. **Download pretrained checkpoints**

Also described in `code/source-code/README.md`:

- Main VA-Count checkpoint.
- Binary classifier checkpoint (optional).
- MAE pretrained backbone.
- YOLO-World weights (if not already present).

---

## **Usage**

All commands below are run from:

```bash
cd code/source-code
```

### 1. Baseline VA-Count training / evaluation

- **Train / fine-tune**:

```bash
python FSC_train.py --data_path ./data/FSC147 \
    --anno_file annotation_FSC147_pos.json \
    --output_dir ./data/out/finetune_pos
```

- **Evaluate** (compute MAE/RMSE):

```bash
python FSC_test.py \
    --data_path ./data/FSC147 \
    --split test \
    --resume ./data/checkpoint_FSC.pth
```

### 2. Generate exemplars with GroundingDINO

```bash
# Positive exemplars
python grounding_pos.py --root_path ./data/FSC147/

# Negative exemplars
python grounding_neg.py --root_path ./data/FSC147/
```

### 3. Generate exemplars with YOLO-World

```bash
# With prompts
python yolo_pos_withPrompt.py --root_path ./data/FSC147/

# Without prompts
python pos_yolo_withoutPrompt.py --root_path ./data/FSC147/

# Negative examples
python yolo_neg.py --root_path ./data/FSC147/
```

### 4. Use Rich Prompts (prompt_enhancer)

Configure your Gemini API key (see `prompt_enhancer.py`), then run the prompt enhancement:

```bash
python prompt_enhancer.py --data_path ./data/FSC147
```

This will generate enhanced prompts that can be consumed by the GroundingDINO / YOLO-World
exemplar-generation scripts.

---

## **Demo Application**

For interactive visualization and inspection of the counting results:

- **Advanced demo with visualization**:

```bash
python demo_app_advanced.py \
    --resume ./data/checkpoint_FSC.pth \
    --data_path ./data/FSC147 \
    --output_dir ./demo_outputs \
    --visualize
```

Additional demo and visualization scripts are available:

- `demo_inference.py`
- `demo_pipeline_advanced.py`
- `demo_visualization.py`

Each script focuses on a specific aspect (pipeline demonstration, visualization, etc.)
and is documented in `code/source-code/README.md`.

---

## **Experimental Results (High-level)**

On **FSC147**, the project observes:

- **Rich Prompt** integration slightly improves baseline VA-Count accuracy (MAE and RMSE).
- **YOLO-World** dramatically reduces exemplar extraction time, at a small cost in raw MAE.
- **YOLO-World + Rich Prompt** achieves a strong **trade-off**:
  - Accuracy comparable to or slightly better than the original VA-Count.
  - Much faster inference suitable for demo and near real-time use.

Exact numbers (MAE, RMSE, runtime) and tables are provided in `Report.pdf`
and can be reproduced from runs archived under `experiments/`.

---

## References

- [1] H. Zhu, S. Li, J. Yuan, Z. Yang, Y. Guo, W. Liu, X. Zhong, and S. He,  
  “Expanding zero-shot object counting with rich prompts,” 2025.  
  [Online]. Available: https://arxiv.org/abs/2505.15398
- [2] H. Zhu, J. Yuan, Z. Yang, Y. Guo, Z. Wang, X. Zhong, and S. He,  
  “Zero-shot object counting with good exemplars,” 2024.  
  [Online]. Available: https://arxiv.org/abs/2407.04948
- [3] C. Liu, Y. Zhong, A. Zisserman, and W. Xie,  
  “Countr: Transformer-based generalised visual counting,” 2022.  
  [Online]. Available: https://arxiv.org/abs/2208.13721
- [4] T. Cheng, L. Song, Y. Ge, W. Liu, X. Wang, and Y. Shan,  
  “Yolo-world: Real-time open-vocabulary object detection,” 2024.  
  [Online]. Available: https://arxiv.org/abs/2401.17270

---

## **Limitations & Future Work**

- In very dense scenes or with heavily overlapping objects, the density map can
  under-count objects.
- For fragmented objects (e.g. windows with many panes, eyeglasses), the model may
  over-count sub-parts as separate instances.
- The MAE + density map paradigm can still produce plausible counts even when
  exemplars fail, indicating some reliance on dataset bias.

Future directions:

- Improve handling of **dense and fragmented objects**.
- Explore alternative counting paradigms beyond density maps (e.g. relational counting).
- Further tighten the coupling between exemplars and prediction to reduce bias.

---

## **License**
This project is developed for **academic purposes** under the course  
**CS338.Q21 – Pattern Recognition** at the **University of Information Technology (UIT)**.

Released under the **MIT License**.
See the [LICENSE](./LICENSE) file for details.
T
