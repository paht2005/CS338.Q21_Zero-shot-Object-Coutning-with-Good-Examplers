### Code directory

This folder contains the full implementation of the project  
“Zero-shot Object Counting with Good Exemplars – Enhanced with Rich Prompts and YOLO-World”.

Current layout:

```text
code/
└── source-code/   # Main VA-Count + Rich Prompt + YOLO-World implementation
```

- **`source-code/`** is a self-contained Python project:
  - Baseline VA-Count (GroundingDINO + MAE + NSM)
  - Rich Prompt pipeline (Gemini-based prompt enhancement + CLIP re-ranking)
  - YOLO-World based exemplar extraction
  - Training, evaluation, and demo scripts

For environment setup, dataset preparation (FSC147), checkpoints, and usage examples,
see `source-code/README.md`.

