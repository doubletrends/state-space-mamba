.PHONY: help install dev test pipeline data features dm train evaluate clean

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

# ── housekeeping ─────────────────────────────────────────────────────────────
clean:          ## remove generated artifacts and caches
	rm -rf output build dist *.egg-info src/*.egg-info .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
