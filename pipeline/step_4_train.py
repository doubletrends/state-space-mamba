"""Stage 4 — model training.

Trains MambaSSM to predict the residual sequence t+1 … t+predict_window.

Input : distinct historical windows anchored at WINDOW_ANCHORS, each WINDOW_LEN days long.
Target: residual[t+1 … t+predict_window]
Loss  : MSE

    Input : config.DM_CSV
    Output: config.CHECKPOINT, config.TRAIN_META_JSON
"""

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from ssm.config import (
    DM_CSV, CHECKPOINT, TRAIN_META_JSON,
    WINDOW_ANCHORS, WINDOW_LEN, PREDICT_WINDOW,
    EPOCHS, PATIENCE, LR, BATCH_SIZE, TARGET_VAL_LOSS, MIN_EPOCHS_BEFORE_TARGET_STOP,
    VAL_SPLIT, VAL_SEED, WEIGHT_DECAY,
    D_MODEL, N_LAYER, D_STATE, DROPOUT,
    ensure_dirs,
)
from ssm.artifacts import load_existing_best, save_checkpoint, write_train_meta
from ssm.data.loader import create_dataloader
from ssm.arch.mamba import create_model


def train():
    ensure_dirs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Task  : predict residual[t+1…t+{PREDICT_WINDOW}]  windows={WINDOW_ANCHORS}×{WINDOW_LEN}\n')

    data = pd.read_csv(DM_CSV, index_col='Date', parse_dates=True)

    print(f'Full data       : {data.index[0].date()} → {data.index[-1].date()}  ({len(data)} rows)')
    split_label = 'chronological' if VAL_SEED is None else f'random (seed={VAL_SEED})'
    print(f'Val split       : {split_label} {1 - VAL_SPLIT:.0%}/{VAL_SPLIT:.0%} on full DM dataset\n')
    print(f'Target stop     : val_loss <= {TARGET_VAL_LOSS:.6f} after epoch {MIN_EPOCHS_BEFORE_TARGET_STOP}\n')

    # Chronological 90/10 train/val split across the full DM dataset.
    train_loader, val_loader = create_dataloader(
        data, WINDOW_ANCHORS, WINDOW_LEN, PREDICT_WINDOW, device,
        val_split=VAL_SPLIT, batch_size=BATCH_SIZE, random_seed=VAL_SEED,
    )

    model     = create_model(data, PREDICT_WINDOW, device, n_windows=len(WINDOW_ANCHORS),
                             d_model=D_MODEL, n_layer=N_LAYER, d_state=D_STATE, dropout=DROPOUT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    best_val_loss, best_epoch, best_source = load_existing_best(TRAIN_META_JSON, CHECKPOINT, data)
    patience_counter = 0

    if best_source:
        print(f'Resuming best    : {best_val_loss:.6f}  (epoch {best_epoch}, from {best_source})')
    else:
        print('Resuming best    : none found; starting fresh')

    pbar = tqdm(range(1, EPOCHS + 1), desc='Training', unit='epoch')
    for epoch in pbar:

        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            pred = model(inputs)
            loss = F.mse_loss(pred, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                pred      = model(inputs)
                val_loss += F.mse_loss(pred, targets).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        scheduler.step(epoch)
        lr = optimizer.param_groups[0]['lr']

        star = '*' if avg_val < best_val_loss else ' '
        pbar.set_postfix({
            'tr_loss': f'{avg_train:.4f}', 'vl_loss': f'{avg_val:.4f}',
            'lr': f'{lr:.1e}', 'pat': patience_counter, 'best': star,
        })

        if avg_val < best_val_loss:
            best_val_loss    = avg_val
            best_epoch       = epoch
            patience_counter = 0
            save_checkpoint(CHECKPOINT, model, data, best_epoch, best_val_loss)
            write_train_meta(TRAIN_META_JSON, data, best_epoch, best_val_loss)
            if epoch >= MIN_EPOCHS_BEFORE_TARGET_STOP and best_val_loss <= TARGET_VAL_LOSS:
                tqdm.write(
                    f'\nReached target val loss {TARGET_VAL_LOSS:.6f} at epoch {epoch} '
                    f'(earliest stop epoch {MIN_EPOCHS_BEFORE_TARGET_STOP}).'
                )
                break
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                tqdm.write(f'\nEarly stopping at epoch {epoch}.')
                break

    write_train_meta(TRAIN_META_JSON, data, best_epoch, best_val_loss)
    print(f'\nBest val loss : {best_val_loss:.6f}  (epoch {best_epoch})')
    print(f'Checkpoint    → {CHECKPOINT}')


if __name__ == '__main__':
    train()
