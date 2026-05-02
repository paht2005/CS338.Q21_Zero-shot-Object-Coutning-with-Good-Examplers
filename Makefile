# Makefile — VA-Count (Zero-shot Object Counting with Good Exemplars)
# CS338.Q21 — Pattern Recognition, UIT

.PHONY: help setup install test eval demo exemplars clean lint

SRC := code/source-code

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## Create conda env and install all dependencies
	bash scripts/setup_env.sh

install: ## Install Python dependencies only (assumes env is active)
	cd $(SRC)/GroundingDINO && pip install -e . && cd ../..
	pip install -r $(SRC)/requirements.txt

data: ## Show instructions for downloading dataset & checkpoints
	bash scripts/download_data.sh

exemplars-dino: ## Generate exemplars using GroundingDINO
	bash scripts/generate_exemplars.sh dino

exemplars-yolo: ## Generate exemplars using YOLO-World
	bash scripts/generate_exemplars.sh yolo

exemplars: ## Generate all exemplars (DINO + YOLO)
	bash scripts/generate_exemplars.sh all

train-baseline: ## Train baseline VA-Count model
	cd $(SRC) && python FSC_pretrain.py \
		--output_dir output/pretrain \
		--data_split_file data/FSC147/Train_Test_Val_FSC_147.json \
		--im_dir data/FSC147/images_384_VarV2 \
		--gt_dir data/FSC147/gt_density_map_adaptive_384_VarV2

eval: ## Run full evaluation suite (baseline + DINO prompt + YOLO)
	bash scripts/run_evaluation.sh

demo: ## Launch Streamlit demo app
	cd $(SRC) && streamlit run demo_app_advanced.py

lint: ## Run basic Python linting
	cd $(SRC) && python -m py_compile models_crossvit.py
	cd $(SRC) && python -m py_compile models_mae_cross.py
	cd $(SRC) && python -m py_compile FSC_train.py
	cd $(SRC) && python -m py_compile FSC_test.py

clean: ## Remove generated outputs and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf $(SRC)/output/test_* 2>/dev/null || true
