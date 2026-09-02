"""Shared artifact IO for training metadata and checkpoints.

This module centralizes the on-disk contract between stage 4 (training) and
stage 5 (evaluation):

    - ``train_meta.json`` schema for dataset/model/window/run configuration
    - checkpoint payload schema for the best model state + best-loss metadata
    - compatibility helpers for older plain ``state_dict`` checkpoints
"""

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ssm.config import (
    BATCH_SIZE,
    D_MODEL,
    DROPOUT,
    D_STATE,
    EPOCHS,
    LR,
    N_LAYER,
    PATIENCE,
    PREDICT_WINDOW,
    MIN_EPOCHS_BEFORE_TARGET_STOP,
    TARGET_VAL_LOSS,
    VAL_SEED,
    VAL_SPLIT,
    WEIGHT_DECAY,
    WINDOW_ANCHORS,
    WINDOW_LEN,
)

META_COMPARE_KEYS = (
    'data_start', 'data_end', 'n_rows', 'window_anchors', 'window_len', 'predict_window',
    'd_model', 'n_layer', 'd_state', 'dropout',
)


def _resume_signature(data: pd.DataFrame) -> dict[str, Any]:
    """Build the subset of train metadata that must match to reuse a prior best."""
    expected = build_train_meta(data, best_epoch=0, best_val_loss=float('inf'))
    return {key: expected[key] for key in META_COMPARE_KEYS}


def build_train_meta(data: pd.DataFrame, best_epoch: int,
                     best_val_loss: float) -> dict[str, Any]:
    """Build the persisted training manifest for the current pipeline contract."""
    return {
        'data_start': data.index[0].strftime('%Y-%m-%d'),
        'data_end': data.index[-1].strftime('%Y-%m-%d'),
        'n_rows': int(len(data)),
        'window_anchors': WINDOW_ANCHORS,
        'window_len': WINDOW_LEN,
        'predict_window': PREDICT_WINDOW,
        'd_model': D_MODEL,
        'n_layer': N_LAYER,
        'd_state': D_STATE,
        'dropout': DROPOUT,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'run_config': {
            'epochs': EPOCHS,
            'patience': PATIENCE,
            'lr': LR,
            'batch_size': BATCH_SIZE,
            'val_split': VAL_SPLIT,
            'val_seed': VAL_SEED,
            'split_mode': 'chronological_full_data' if VAL_SEED is None else 'random_full_data',
            'target_val_loss': TARGET_VAL_LOSS,
            'min_epochs_before_target_stop': MIN_EPOCHS_BEFORE_TARGET_STOP,
            'weight_decay': WEIGHT_DECAY,
        },
    }


def write_train_meta(path: Path, data: pd.DataFrame, best_epoch: int,
                     best_val_loss: float) -> None:
    """Persist the current training manifest."""
    with path.open('w') as f:
        json.dump(build_train_meta(data, best_epoch, best_val_loss), f, indent=2)


def load_train_meta(path: Path) -> dict[str, Any]:
    """Load the persisted training manifest."""
    with path.open() as f:
        return json.load(f)


def checkpoint_payload(model: torch.nn.Module, data: pd.DataFrame,
                       best_epoch: int, best_val_loss: float) -> dict[str, Any]:
    """Build the checkpoint payload saved by stage 4."""
    return {
        'model_state_dict': model.state_dict(),
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'resume_signature': _resume_signature(data),
    }


def save_checkpoint(path: Path, model: torch.nn.Module, data: pd.DataFrame,
                    best_epoch: int, best_val_loss: float) -> None:
    """Save the best checkpoint in the shared payload format."""
    torch.save(checkpoint_payload(model, data, best_epoch, best_val_loss), path)


def load_checkpoint_state(path: Path, device: torch.device | str) -> dict[str, Any]:
    """Load model weights from either a checkpoint payload or an old plain state dict."""
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and 'model_state_dict' in payload:
        return payload['model_state_dict']
    return payload


def load_existing_best(meta_path: Path, checkpoint_path: Path,
                       data: pd.DataFrame) -> tuple[float, int, str | None]:
    """Recover the current best loss/epoch from matching meta or checkpoint payload.

    The metadata is only trusted when it matches the current window/model contract.
    Older plain ``state_dict`` checkpoints intentionally do not provide loss metadata.
    """
    expected_signature = _resume_signature(data)

    if meta_path.exists():
        meta = load_train_meta(meta_path)
        if all(meta.get(key) == expected_signature[key] for key in META_COMPARE_KEYS):
            best_val_loss = meta.get('best_val_loss')
            best_epoch = meta.get('best_epoch', 0)
            if isinstance(best_val_loss, (int, float)) and math.isfinite(best_val_loss):
                return float(best_val_loss), int(best_epoch), 'train_meta.json'

    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(payload, dict) and 'model_state_dict' in payload:
            signature = payload.get('resume_signature')
            if signature != expected_signature:
                return float('inf'), 0, None
            best_val_loss = payload.get('best_val_loss')
            best_epoch = payload.get('best_epoch', 0)
            if isinstance(best_val_loss, (int, float)) and math.isfinite(best_val_loss):
                return float(best_val_loss), int(best_epoch), 'checkpoint payload'

    return float('inf'), 0, None
