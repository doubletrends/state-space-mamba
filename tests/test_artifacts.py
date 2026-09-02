"""Artifact IO must preserve the current train/eval contract."""

import torch

from ssm import artifacts


def test_load_existing_best_prefers_matching_meta(tmp_path):
    meta_path = tmp_path / 'train_meta.json'
    ckpt_path = tmp_path / 'best.pt'
    best_loss = 0.123
    idx = artifacts.pd.date_range('2025-01-01', periods=3, freq='D')
    data = artifacts.pd.DataFrame({'x': [1.0, 2.0, 3.0]}, index=idx)

    expected = artifacts.build_train_meta(
        data=data,
        best_epoch=17,
        best_val_loss=best_loss,
    )
    artifacts.write_train_meta(meta_path, data, 17, best_loss)
    torch.save({
        'model_state_dict': {},
        'best_epoch': 99,
        'best_val_loss': 9.99,
        'resume_signature': artifacts._resume_signature(data),
    }, ckpt_path)

    found_loss, found_epoch, source = artifacts.load_existing_best(meta_path, ckpt_path, data)
    assert found_loss == best_loss
    assert found_epoch == 17
    assert source == 'train_meta.json'
    assert artifacts.load_train_meta(meta_path) == expected


def test_load_existing_best_falls_back_to_matching_checkpoint_payload(tmp_path):
    meta_path = tmp_path / 'train_meta.json'
    ckpt_path = tmp_path / 'best.pt'
    idx = artifacts.pd.date_range('2025-01-01', periods=3, freq='D')
    data = artifacts.pd.DataFrame({'x': [1.0, 2.0, 3.0]}, index=idx)

    meta_path.write_text('{"data_end":"1999-01-01","best_epoch":1,"best_val_loss":1.0}')
    torch.save({
        'model_state_dict': {},
        'best_epoch': 23,
        'best_val_loss': 0.456,
        'resume_signature': artifacts._resume_signature(data),
    }, ckpt_path)

    found_loss, found_epoch, source = artifacts.load_existing_best(meta_path, ckpt_path, data)
    assert found_loss == 0.456
    assert found_epoch == 23
    assert source == 'checkpoint payload'


def test_load_existing_best_rejects_stale_checkpoint_payload(tmp_path):
    meta_path = tmp_path / 'train_meta.json'
    ckpt_path = tmp_path / 'best.pt'
    idx = artifacts.pd.date_range('2025-01-01', periods=3, freq='D')
    data = artifacts.pd.DataFrame({'x': [1.0, 2.0, 3.0]}, index=idx)
    stale = artifacts.pd.DataFrame({'x': [1.0, 2.0]}, index=idx[:2])

    meta_path.write_text('{"data_end":"1999-01-01","best_epoch":1,"best_val_loss":1.0}')
    torch.save({
        'model_state_dict': {},
        'best_epoch': 23,
        'best_val_loss': 0.456,
        'resume_signature': artifacts._resume_signature(stale),
    }, ckpt_path)

    found_loss, found_epoch, source = artifacts.load_existing_best(meta_path, ckpt_path, data)
    assert found_loss == float('inf')
    assert found_epoch == 0
    assert source is None


def test_load_checkpoint_state_supports_payload_and_plain_state_dict(tmp_path):
    ckpt_payload = tmp_path / 'payload.pt'
    ckpt_plain = tmp_path / 'plain.pt'
    expected = {'layer.weight': torch.ones(2, 2)}

    idx = artifacts.pd.date_range('2025-01-01', periods=3, freq='D')
    data = artifacts.pd.DataFrame({'x': [1.0, 2.0, 3.0]}, index=idx)

    torch.save({
        'model_state_dict': expected,
        'best_epoch': 1,
        'best_val_loss': 0.1,
        'resume_signature': artifacts._resume_signature(data),
    }, ckpt_payload)
    torch.save(expected, ckpt_plain)

    assert artifacts.load_checkpoint_state(ckpt_payload, 'cpu').keys() == expected.keys()
    assert artifacts.load_checkpoint_state(ckpt_plain, 'cpu').keys() == expected.keys()
