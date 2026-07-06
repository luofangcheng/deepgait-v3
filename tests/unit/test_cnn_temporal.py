"""Unit tests for the temporal 1D-CNN model (Category C of the
experiment).

These tests verify the model architecture is well-formed and that
training/inference loops work end-to-end on synthetic data — without
requiring real fTIR data. The CNN trains on small synthetic examples
and is then evaluated on a holdout split.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from deepgait3.core.pawprint.cnn_temporal import (
    TemporalFootprintCNN,
    predict_temporal,
    train_temporal_cnn,
)


@pytest.fixture
def synthetic_data():
    """Tiny synthetic dataset: 200 samples, 8 time steps, binary labels."""
    rng = np.random.default_rng(42)
    T = 8
    n = 200
    X = rng.normal(0, 1, (n, T)).astype(np.float32)
    y = (X[:, T // 2] > 0).astype(np.float32)  # simple rule
    y_full = np.zeros((n, T), dtype=np.float32)
    y_full[:, T // 2] = y
    return X, y_full


def test_model_forward_shape():
    """Forward pass returns (B, T) logits."""
    model = TemporalFootprintCNN(time_steps=8)
    x = torch.randn(4, 8)
    out = model(x)
    assert out.shape == (4, 8)


def test_model_single_input():
    """Model accepts a single sample (B=1)."""
    model = TemporalFootprintCNN(time_steps=8)
    x = torch.randn(1, 8)
    out = model(x)
    assert out.shape == (1, 8)


def test_train_decreases_loss():
    """After training, the loss should decrease from the initial value."""
    X = np.random.default_rng(0).normal(0, 1, (100, 8)).astype(np.float32)
    y = np.zeros((100, 8), dtype=np.float32)
    y[:, 4] = (X[:, 4] > 0).astype(np.float32)
    X_tr, y_tr = X[:80], y[:80]
    X_va, y_va = X[80:], y[80:]
    model, history = train_temporal_cnn(
        X_tr, y_tr, X_va, y_va,
        epochs=20, batch_size=16, lr=1e-2, verbose=False,
    )
    first_loss = history["train_loss"][0]
    last_loss = history["train_loss"][-1]
    assert last_loss < first_loss, f"Loss did not decrease: {first_loss} -> {last_loss}"


def test_predict_returns_probabilities():
    """predict_temporal returns values in [0, 1]."""
    X = np.random.default_rng(0).normal(0, 1, (50, 8)).astype(np.float32)
    y = np.zeros((50, 8), dtype=np.float32)
    model, _ = train_temporal_cnn(X, y, X[:10], y[:10],
                                   epochs=5, verbose=False)
    probs = predict_temporal(model, X)
    assert probs.shape == (50, 8)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_cnn_learns_simple_pattern():
    """CNN trained on a synthetic pattern should beat random chance."""
    rng = np.random.default_rng(123)
    n = 500
    T = 8
    X = rng.normal(0, 1, (n, T)).astype(np.float32)
    # Label = 1 if sum at center > sum at edges
    y_full = np.zeros((n, T), dtype=np.float32)
    y_full[:, T // 2] = (X[:, T // 2] > 0.5).astype(np.float32)
    X_tr, y_tr = X[:400], y_full[:400]
    X_va, y_va = X[400:], y_full[400:]
    model, history = train_temporal_cnn(
        X_tr, y_tr, X_va, y_va,
        epochs=30, batch_size=64, lr=1e-2, verbose=False,
    )
    # Val accuracy should be > 0.7 (random would be ~0.5 for balanced)
    assert history["val_acc"][-1] > 0.7, \
        f"CNN val_acc too low: {history['val_acc'][-1]}"