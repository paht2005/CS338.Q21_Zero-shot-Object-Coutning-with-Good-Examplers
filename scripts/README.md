# `scripts/`

Reserved for **shell / Python helper scripts** that wrap common workflows
(dataset download, full evaluation sweeps, log post-processing). The
CS338.Q21 iteration kept all execution at the Python module level under
`code/source-code/`, so this folder is currently empty.

Suggested layout if scripts are added in a future iteration:

```
scripts/
├── prepare_fsc147.sh        # Download + unzip FSC-147 into code/source-code/data/
├── run_full_eval.sh         # Run FSC_test.py for all four configurations
└── extract_wandb_metrics.py # Build RESULTS.md tables straight from wandb summaries
```

Anything that produces numbers cited in the report should be reproducible
from a single command in this folder.
