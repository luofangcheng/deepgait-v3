"""Temporal 1D-CNN for footprint contact detection.

Inspired by InterDigitalInc/UnderPressure (SIGGRAPH 2022):
"Deep Learning for Foot Contact Detection, Ground Reaction Force Estimation".

Our adaptation for fTIR:
- Input:  T × H × W per-frame MExG score maps (T=8 frames)
- Output: T per-frame binary footprint masks
- Training data: 25 GT prints × multiple frames (with augmentation)
- GPU: trained on CUDA if available, CPU fallback for inference

Architecture (simplified from UnderPressure):
  Conv1D (kernel=5, channels 1→32→64→128) → mean+max pool
  → Linear (256 → T)
  → sigmoid per frame
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalFootprintCNN(nn.Module):
    """1D-CNN over a temporal window of MExG score maps.

    Per-pixel classifier: given T frames of score map at pixel (h, w),
    predict whether pixel (h, w) is part of a footprint at each of those
    T frames.

    We pool over the spatial dim by extracting one pixel's T-frame
    time series, then classify per-pixel.
    """

    def __init__(self, time_steps: int = 8, channels: int = 1,
                  hidden: int = 64):
        super().__init__()
        self.time_steps = time_steps
        # Conv1D over time: input (B, channels, T) → (B, hidden, T)
        self.conv1 = nn.Conv1d(channels, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden * 2, hidden, kernel_size=3, padding=1)
        # Per-frame head: (B, hidden, T) → (B, 1, T)
        self.head = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, time_steps) → (B, time_steps) logits."""
        # Add channel dim: (B, T) → (B, 1, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = self.head(h)              # (B, 1, T)
        h = h.squeeze(1)              # (B, T)
        return h


def train_temporal_cnn(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: Optional[str] = None,
    verbose: bool = True,
) -> tuple[nn.Module, dict]:
    """Train the temporal CNN.

    X: (N, time_steps) — per-pixel time series of MExG scores
    y: (N, time_steps) — per-pixel binary labels (1=footprint, 0=bg)

    Returns: (trained model, training_history dict)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"Training on {device}")

    model = TemporalFootprintCNN(time_steps=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = torch.tensor([5.0]).to(device)  # footprints are rare
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_tr_t = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_tr_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    X_va_t = torch.from_numpy(X_val.astype(np.float32)).to(device)
    y_va_t = torch.from_numpy(y_val.astype(np.float32)).to(device)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    n = X_tr_t.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_tr_t[idx]
            yb = y_tr_t[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_train = total_loss / n

        # Val
        model.eval()
        with torch.no_grad():
            val_logits = model(X_va_t)
            val_loss = criterion(val_logits, y_va_t).item()
            val_pred = (torch.sigmoid(val_logits) > 0.5).float()
            val_acc = (val_pred == y_va_t).float().mean().item()
        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:>3}: train_loss={avg_train:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    return model, history


def predict_temporal(model: nn.Module, X: np.ndarray,
                      device: Optional[str] = None,
                      batch_size: int = 4096) -> np.ndarray:
    """Predict per-frame footprint probability for each pixel.

    Returns: (N, time_steps) probability array in [0, 1].
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    out = []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch_size):
            xb = X_t[i:i + batch_size]
            logits = model(xb)
            prob = torch.sigmoid(logits).cpu().numpy()
            out.append(prob)
    return np.concatenate(out, axis=0)


__all__ = [
    "TemporalFootprintCNN",
    "train_temporal_cnn",
    "predict_temporal",
]