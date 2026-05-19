from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.schedule import DDPMSchedule
from utils.tools import EarlyStopping


def _pretrain_epoch(
    model: nn.Module,
    loader: DataLoader,
    schedule: DDPMSchedule,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    device: str,
) -> float:
    model.train()
    total = 0.0
    for t, X, data_mask, _ in loader:
        t = t.to(device)
        X = X.to(device)
        data_mask = data_mask.to(device)
        B = X.shape[0]
        s = torch.randint(0, schedule.alpha_bars.shape[0], (B,), device=device)
        X_noisy, eps = schedule.q_sample(X, s, data_mask)
        eps_pred = model.denoise(t, X_noisy, data_mask, s)
        loss = ((eps_pred - eps) ** 2 * data_mask).sum() / data_mask.sum().clamp(min=1)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def _val_epoch(
    model: nn.Module,
    loader: DataLoader,
    schedule: DDPMSchedule,
    device: str,
) -> float:
    model.eval()
    total = 0.0
    for t, X, data_mask, _ in loader:
        t = t.to(device)
        X = X.to(device)
        data_mask = data_mask.to(device)
        B = X.shape[0]
        s = torch.randint(0, schedule.alpha_bars.shape[0], (B,), device=device)
        X_noisy, eps = schedule.q_sample(X, s, data_mask)
        eps_pred = model.denoise(t, X_noisy, data_mask, s)
        loss = ((eps_pred - eps) ** 2 * data_mask).sum() / data_mask.sum().clamp(min=1)
        total += loss.item()
    return total / len(loader)


def pretrain(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    schedule: DDPMSchedule,
    max_epochs: int,
    lr: float,
    patience: int,
    checkpoint_path: str,
    grad_clip: float = 1.0,
    device: str = "cpu",
) -> float:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    es = EarlyStopping(patience=patience, path=checkpoint_path, verbose=True)

    for epoch in range(max_epochs):
        train_loss = _pretrain_epoch(model, train_loader, schedule, optimizer, grad_clip, device)
        val_loss = _val_epoch(model, val_loader, schedule, device)
        print(f"Pretrain epoch {epoch+1}/{max_epochs}  train={train_loss:.4f}  val={val_loss:.4f}")
        es(val_loss)
        if es.counter == 0:
            es.save_checkpoint(val_loss, model)
        if es.early_stop:
            print("Early stopping.")
            break

    return es.val_loss_min
