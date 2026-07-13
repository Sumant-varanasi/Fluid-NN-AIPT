"""Shared training/evaluation harness so every equalizer is compared on identical
data, optimizer budget, and metrics."""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fluidnn.channel.modulation import QAM
from fluidnn.metrics import equalizer_report


def train_equalizer(
    model: nn.Module,
    x_train: np.ndarray,  # (n, T, 2) float32
    y_train: np.ndarray,  # (n, 2) float32
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
    lr_min: float | None = None,  # if set, cosine-anneal lr -> lr_min over the run
    weight_decay: float = 0.0,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    torch.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    xv = torch.from_numpy(x_val)
    yv = torch.from_numpy(y_val)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr_min)
        if lr_min is not None
        else None
    )
    loss_fn = nn.MSELoss()

    history = {"train_mse": [], "val_mse": []}
    best_val = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        running, count = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
            count += len(xb)
        if scheduler is not None:
            scheduler.step()
        model.eval()
        with torch.no_grad():
            val_mse = loss_fn(_predict(model, xv), yv).item()
        history["train_mse"].append(running / count)
        history["val_mse"].append(val_mse)
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose:
            print(
                f"  epoch {epoch + 1:3d}/{epochs}  train MSE {running / count:.5f}"
                f"  val MSE {val_mse:.5f}  ({time.time() - t0:.0f}s)"
            )
    model.load_state_dict(best_state)
    history["train_seconds"] = time.time() - t0
    return history


def _predict(model: nn.Module, x: torch.Tensor, batch: int = 4096) -> torch.Tensor:
    outs = [model(x[i : i + batch]) for i in range(0, len(x), batch)]
    return torch.cat(outs)


def evaluate_equalizer(
    model: nn.Module,
    x_test: np.ndarray,
    tx_symbols: np.ndarray,
    tx_bits: np.ndarray,
    qam: QAM,
) -> dict:
    model.eval()
    with torch.no_grad():
        pred = _predict(model, torch.from_numpy(x_test)).numpy()
    eq = pred[:, 0] + 1j * pred[:, 1]
    report = equalizer_report(eq, tx_symbols, tx_bits, qam)
    report["params"] = sum(p.numel() for p in model.parameters())
    report["macs_per_symbol"] = model.macs_per_symbol()
    return report
