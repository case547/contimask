from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from utils.tools import EarlyStopping


def _build_optimizer(
    model: nn.Module,
    lr_ratios: list[float],
    base_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    embedding_params = (
        list(model.input_proj.parameters())
        + list(model.time_embed.parameters())
        + list(model.step_embed.parameters())
        + [model.cls_token]
    )
    param_groups = [{"params": embedding_params, "lr": base_lr * lr_ratios[0]}]
    for i, layer in enumerate(model.transformer.layers):
        ratio = lr_ratios[i]
        if ratio == 0.0:
            for p in layer.parameters():
                p.requires_grad_(False)
        else:
            param_groups.append({"params": list(layer.parameters()), "lr": base_lr * ratio})
    param_groups.append({"params": list(model.cls_head.parameters()), "lr": base_lr})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def _finetune_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    pos_weight: torch.Tensor,
    grad_clip: float,
    device: str,
) -> float:
    model.train()
    total = 0.0
    for t, X, dm, labels in loader:
        t, X, dm = t.to(device), X.to(device), dm.to(device)
        labels = labels.float().to(device)
        logit = model.classify(t, X, dm).squeeze(-1)  # (B,)
        loss = F.binary_cross_entropy_with_logits(
            logit, labels, pos_weight=pos_weight.to(device)
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    all_preds, all_labels = [], []
    for t, X, dm, labels in loader:
        t, X, dm = t.to(device), X.to(device), dm.to(device)
        prob = torch.sigmoid(model.classify(t, X, dm).squeeze(-1)).cpu()
        all_preds.append(prob)
        all_labels.append(labels)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    if labels.sum() == 0:
        return 0.0
    return float(average_precision_score(labels, preds))


def finetune(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr_ratios: list[float],
    base_lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    checkpoint_path: str,
    grad_clip: float = 1.0,
    device: str = "cpu",
) -> float:
    model.to(device)

    # Compute pos_weight from training labels
    all_labels = torch.cat([labels for _, _, _, labels in train_loader])
    n_pos = all_labels.sum().item()
    n_neg = len(all_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)])

    optimizer = _build_optimizer(model, lr_ratios, base_lr, weight_decay)
    es = EarlyStopping(patience=patience, path=checkpoint_path, verbose=True)

    best_auprc = 0.0
    for epoch in range(max_epochs):
        _finetune_epoch(model, train_loader, optimizer, pos_weight, grad_clip, device)
        val_auprc = _evaluate(model, val_loader, device)
        print(f"Finetune epoch {epoch+1}/{max_epochs}  val_auprc={val_auprc:.4f}")
        es(-val_auprc)  # EarlyStopping minimises; negate AUPRC so higher = better
        if es.counter == 0:
            es.save_checkpoint(-val_auprc, model)
            best_auprc = val_auprc
        if es.early_stop:
            print("Early stopping.")
            break

    return best_auprc
