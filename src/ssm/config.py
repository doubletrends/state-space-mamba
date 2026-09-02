"""Central configuration — the contract shared *between* pipeline stages.

This module is the single source of truth for everything that crosses a stage
boundary: directory locations, the artifact filenames each stage hands to the
next, and the horizon / model / training hyperparameters used by more than one
stage.

Stage-*internal* structural config stays in its own stage on purpose — e.g. the
feature registry in ``step_3_dm`` and the oscillator/detrend tables in
``step_2_feature_engineering``. Those are local to one stage's logic, not a knob
shared across the pipeline, so centralising them here would only hurt locality.

Pipeline data flow (each stage reads the previous stage's artifacts)::

    step_1_data_ingestion      ──▶  ODS_CSV
    step_2_feature_engineering ──▶  DWD_CSV, TREND_PARAMS_JSON (reads ODS_CSV)
    step_3_dm                  ──▶  DM_CSV, NORM_PARAMS_JSON   (reads DWD_CSV)
    step_4_train               ──▶  CHECKPOINT, TRAIN_META_JSON (reads DM_CSV)
    step_5_evaluate            ──▶  future-rollout figures + csv      (reads all of the above)
"""

from pathlib import Path

import pandas as pd

# ── directories ──────────────────────────────────────────────────────────────
# config.py lives at <root>/src/ssm/config.py, so the repo root is parents[2].
ROOT           = Path(__file__).resolve().parents[2]
OUTPUT         = ROOT / "output"
CHECKPOINT_DIR = OUTPUT / "checkpoints"

# ── stage artifacts (the inter-stage contract) ───────────────────────────────
ODS_CSV           = OUTPUT / "step1_ods.csv"
DWD_CSV           = OUTPUT / "step2_dwd.csv"
DM_CSV            = OUTPUT / "step3_dm.csv"
TREND_PARAMS_JSON = OUTPUT / "trend_params.json"
NORM_PARAMS_JSON  = OUTPUT / "norm_params.json"
CHECKPOINT        = CHECKPOINT_DIR / "MambaSSM_best.pt"
TRAIN_META_JSON   = OUTPUT / "train_meta.json"

# ── windowing / forecast contract shared by train + evaluate ─────────────────
FORECAST_DAYS  = 365   # synthetic future rollout horizon after the last observed date

# Sequential look-back windows: each anchor below becomes a WINDOW_LEN-day *consecutive*
# window (days [a .. a+WINDOW_LEN-1] before t). Windows are kept DISTINCT — the model
# encodes each window's short sequence with a SHARED Mamba and concatenates the per-window
# embeddings, so the calendar gaps between windows never enter a shared recurrence.
WINDOW_ANCHORS = [365, 180, 90, 75, 60, 45, 30]   # nearest day of each window
WINDOW_LEN     = 7

PREDICT_WINDOW = 7     # predict the residual sequence t+1 … t+7 jointly

# ── domain constants ─────────────────────────────────────────────────────────
# All known + estimated future halvings. Used by step_2 (historical features)
# and step_5 (rollout). Including future dates is safe for historical lookups —
# they are never selected because no observed date reaches them yet.
HALVING_DATES = pd.to_datetime([
    '2012-11-28',
    '2016-07-09',
    '2020-05-11',
    '2024-04-20',
    '2028-04-20',   # estimated
])

# ── training hyperparameters ─────────────────────────────────────────────────
EPOCHS       = 1000
PATIENCE     = 1000
LR           = 3e-4
BATCH_SIZE   = 32
VAL_SPLIT    = 0.10   # approx. 370 target days out of the current full window set
VAL_SEED     = 42     # fixed random split over the full DM dataset
TARGET_VAL_LOSS = 0.015
MIN_EPOCHS_BEFORE_TARGET_STOP = 50
WEIGHT_DECAY = 1e-3

# ── model hyperparameters ────────────────────────────────────────────────────
D_MODEL  = 32
N_LAYER  = 3
D_STATE  = 16
DROPOUT  = 0.1


def ensure_dirs() -> None:
    """Create the output directories used by the pipeline if they don't exist."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
