.PHONY: help install dev test pipeline data features dm train evaluate exp1 exp1-sanity exp2 experiments figures clean

.DEFAULT_GOAL := help

# ── help ─────────────────────────────────────────────────────────────────────
help:           ## list available targets
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## /{printf "  %-9s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ── setup ────────────────────────────────────────────────────────────────────
install:        ## editable install of the ssm package
	pip install -e .

dev:            ## install with test dependencies
	pip install -e ".[dev]"

# ── quality ──────────────────────────────────────────────────────────────────
test:           ## run the test suite
	pytest -q

# ── pipeline stages ──────────────────────────────────────────────────────────
data:           ## stage 1 — data ingestion
	python pipeline/step_1_data_ingestion.py

features:       ## stage 2 — feature engineering
	python pipeline/step_2_feature_engineering.py

dm:             ## stage 3 — DM assembly
	python pipeline/step_3_dm.py

train:          ## stage 4 — train
	python pipeline/step_4_train.py

evaluate:       ## stage 5 — evaluate
	python pipeline/step_5_evaluate.py

pipeline: data features dm train evaluate  ## run all five stages in order

# ── experiments ───────────────────────────────────────────────────────────────
exp1:           ## control — Naive Bayes vs learned combiners, walk-forward
	python experiments/exp_1_control.py

exp1-sanity:    ## encoder fidelity: in-sample skill + probe reproduction
	python experiments/exp_1_sanity.py

exp2:           ## MambaSSM refit as a directional classifier, same protocol
	python experiments/exp_2_direction.py

experiments: exp1 exp2  ## run both walk-forward experiments

figures:        ## regenerate the README figures from the scored summaries
	python experiments/plot_architecture.py
	python experiments/plot_summary.py

# ── housekeeping ─────────────────────────────────────────────────────────────
clean:          ## remove generated artifacts and caches
	rm -rf output build dist *.egg-info src/*.egg-info .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
